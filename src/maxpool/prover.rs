//! To prove a maxpooling layer:
//! Input:
//!   1. A vector of scalars `y1`, represented as a multivariate polynomial `y1(b1, b2, ..., bn, c1, ..., cm)`.
//!   2. Maxpooling result `y2`, represented as a multivariate polynomial `y2(b3, b4, ..., bn, c1, ..., cm)`.
//!     * `y1` and `y2` has already been reordered compared to the original input in dat file.
//!
//! Proof process:
//!   1. Compute the partial evaluations of `y1` for fixed values of `(b1, b2)`:
//!      `y1(0, 0, b3, ..., bn, c1, ..., cm)`, `y1(0, 1, b3, ..., bn, c1, ..., cm)`,
//!      `y1(1, 0, b3, ..., bn, c1, ..., cm)`, `y1(1, 1, b3, ..., bn, c1, ..., cm)`.
//!   2. Use the sumcheck protocol to prove the following equation:
//!         ∑_{b3, b4, ..., bn, c1, ..., cm} (y2(b3, b4, ..., bn, c1, ..., cm)
//!                                          - y1(0, 0, b3, ..., bn, c1, ..., cm))
//!                                          * (y2(b3, b4, ..., bn, c1, ..., cm)
//!                                          - y1(0, 1, b3, ..., bn, c1, ..., cm))
//!                                          * (y2(b3, b4, ..., bn, c1, ..., cm)
//!                                          - y1(1, 0, b3, ..., bn, c1, ..., cm))
//!                                          * (y2(b3, b4, ..., bn, c1, ..., cm)
//!                                          - y1(1, 1, b3, ..., bn, c1, ..., cm)) = 0.
//!   3. Use the logup protocol to prove the inequality constraints:
//!         y2(b3, b4, ..., bn, c1, ..., cm) >= y1(0, 0, b3, ..., bn, c1, ..., cm),
//!         y2(b3, b4, ..., bn, c1, ..., cm) >= y1(0, 1, b3, ..., bn, c1, ..., cm),
//!         y2(b3, b4, ..., bn, c1, ..., cm) >= y1(1, 0, b3, ..., bn, c1, ..., cm),
//!         y2(b3, b4, ..., bn, c1, ..., cm) >= y1(1, 1, b3, ..., bn, c1, ..., cm).
//!      This can be expressed as:
//!         f = y2(b3, b4, ..., bn, c1, ..., cm) - y1(b1, b2, ..., bn, c1, ..., cm) >= 0.

use crate::{E, F};
use ark_ec::pairing::Pairing;
use ark_ff::{One, Zero};
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::vec::Vec;
use ark_sumcheck::ml_sumcheck::{
    data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck, Proof as SumcheckProof,
};
use logup::{Logup, LogupProof};
use merlin::Transcript;
use pcs::multilinear_kzg::data_structures::{MultilinearProverParam, MultilinearVerifierParam};

const MAX_VALUE_IN_Y: u64 = 65536;

pub struct Prover {
    pub y1: Rc<DenseMultilinearExtension<F>>, // Use F instead of Fr
    pub y2: Rc<DenseMultilinearExtension<F>>,
    pub num_vars_y1: usize,
    pub num_vars_y2: usize,
}

/// Reorders groups of variables in a multivariate polynomial's evaluations vector based on a new order.
///
/// # Purpose
/// This function is designed to reorder the variable groups in a multivariate polynomial's evaluation table.
/// For example, if the original polynomial is represented as `f1(channel, (num_vars_in_data - 2) / 2, b0, (num_vars_in_data - 2) / 2, b1)`
/// and the goal is to reorder it to `f1(b0, b1, (num_vars_in_data - 2) / 2, (num_vars_in_data - 2) / 2, channel)`,
/// this function enables that reordering.
///
/// # Parameters
/// - `poly`: A reference to the `DenseMultilinearExtension<F>` object representing the polynomial.
///           This includes the evaluations of the polynomial over all variable assignments.
/// - `group_sizes`: A slice where each entry specifies the number of variables in each group,
///                  ordered from least significant to most significant in the original polynomial.
///                  For example, `[1, (num_vars_in_data - 2) / 2, 1, (num_vars_in_data - 2) / 2, num_vars_in_channel]`.
/// - `new_order`: A slice specifying the new order of variable groups, from least significant to most significant
///                after reordering. Each entry is an index into the original `group_sizes` array.
///                For example, `[4, 1, 3, 0, 2]` means:
///                - The group at index 4 (channel) in the original `group_sizes` array becomes the least significant
///                  group in the reordered polynomial.
///                - The group at index 1 (`(num_vars_in_data - 2) / 2`) in the original `group_sizes` array becomes
///                  the second least significant group.
///                - The group at index 3 (`(num_vars_in_data - 2) / 2`) in the original `group_sizes` array becomes
///                  the third least significant group.
///                - The group at index 0 (`b1`) in the original `group_sizes` array becomes the fourth least
///                  significant group.
///                - The group at index 2 (`b0`) in the original `group_sizes` array becomes the most significant
///                  group.
///                After reordering, the resulting polynomial's variable arrangement will match the specified order:
///                `f1(b0, b1, (num_vars_in_data - 2) / 2, (num_vars_in_data - 2) / 2, channel)`.
///
pub fn reorder_variable_groups(
    poly: &DenseMultilinearExtension<F>,
    group_sizes: &[usize],
    new_order: &[usize],
) -> DenseMultilinearExtension<F> {
    // 1) 算出每个 group 在旧顺序下的 bit 起始偏移 (从低位到高位)
    let mut original_offsets = Vec::with_capacity(group_sizes.len());
    let mut acc = 0;
    for &size in group_sizes {
        original_offsets.push(acc);
        acc += size;
    }
    let num_vars = poly.num_vars;
    assert_eq!(acc, num_vars, "sum of group_sizes must equal num_vars");

    // 2) 算出每个 group 在新顺序下的 bit 起始偏移 (同样从低位到高位)
    let mut new_group_offsets = Vec::with_capacity(group_sizes.len());
    let mut cur = 0;
    for &g in new_order {
        new_group_offsets.push(cur);
        cur += group_sizes[g];
    }

    // 3) var_map: 老的“变量下标” -> 新的“变量下标”
    //    注意，这里 “变量下标” 就是从 0..(num_vars-1)，且 0 是最低位，num_vars-1 是最高位
    let mut var_map = vec![0; num_vars];
    for (new_gpos, &old_gpos) in new_order.iter().enumerate() {
        let old_offset = original_offsets[old_gpos];
        let size_old = group_sizes[old_gpos];
        let new_offset = new_group_offsets[new_gpos];
        for k in 0..size_old {
            var_map[old_offset + k] = new_offset + k;
        }
    }

    // 4) 用 “显式提取 bits + 重排 bits + 重组 bits” 的方式构造 new_evals
    let size = 1 << num_vars;
    let mut new_evals = vec![F::default(); size];
    for old_index in 0..size {
        // a) 从 old_index 提取出各个 bit (b1,b2,b3,b4,...)，
        //    其中 bit i = (old_index >> i) & 1 （i=0是最低位）
        let mut old_bits = vec![0u8; num_vars];
        for i in 0..num_vars {
            old_bits[i] = ((old_index >> i) & 1) as u8;
        }

        // b) 根据 var_map，把 old_bits[i] 放到 new_bits[var_map[i]]
        let mut new_bits = vec![0u8; num_vars];
        for i in 0..num_vars {
            let j = var_map[i];
            new_bits[j] = old_bits[i];
        }

        // c) 把 new_bits 拼回一个新的索引 (依然是 new_bits[0] 做最低位)
        let mut new_index = 0usize;
        for i in 0..num_vars {
            let bit_val = (new_bits[i] as usize) & 1;
            new_index |= bit_val << i;
        }

        // d) 赋值
        new_evals[new_index] = poly.evaluations[old_index];
    }

    DenseMultilinearExtension::from_evaluations_vec(num_vars, new_evals)
}

impl Prover {
    pub fn new(
        y1: Rc<DenseMultilinearExtension<F>>,
        y2: Rc<DenseMultilinearExtension<F>>,
        num_vars_y1: usize,
        num_vars_y2: usize,
    ) -> Self {
        Self {
            y1,
            y2,
            num_vars_y1,
            num_vars_y2,
        }
    }

    /// Partial evaluation of y1 at fixed values of (b1, b2).
    fn partial_eval_y1(&self, b1: bool, b2: bool) -> Rc<DenseMultilinearExtension<F>> {
        let n = self.num_vars_y1;
        let sub_n = n - 2;
        let size = 1 << n;
        let sub_size = 1 << sub_n;

        let mut sub_evals = Vec::with_capacity(sub_size);
        for sub_i in 0..sub_size {
            let i = ((b1 as usize) << (n - 1)) | ((b2 as usize) << (n - 2)) | sub_i;
            sub_evals.push(self.y1.evaluations[i]);
        }

        Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            sub_n, sub_evals,
        ))
    }

    /// Evaluate y1 slices for (b1, b2) = (0, 0), (0, 1), (1, 0), (1, 1).
    fn evaluate_y1_slices(
        &self,
    ) -> (
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
        Rc<DenseMultilinearExtension<F>>,
    ) {
        let y1_00 = self.partial_eval_y1(false, false);
        let y1_01 = self.partial_eval_y1(false, true);
        let y1_10 = self.partial_eval_y1(true, false);
        let y1_11 = self.partial_eval_y1(true, true);
        (y1_00, y1_01, y1_10, y1_11)
    }

    /// Compute the difference y2 - y1_xx for each x in the domain.
    fn diff_mle(
        a: &Rc<DenseMultilinearExtension<F>>,
        b: &Rc<DenseMultilinearExtension<F>>,
    ) -> Rc<DenseMultilinearExtension<F>> {
        let nv = a.num_vars;
        assert_eq!(nv, b.num_vars);
        let mut vals = Vec::with_capacity(1 << nv);
        for i in 0..(1 << nv) {
            vals.push(a.evaluations[i] - b.evaluations[i]);
        }
        Rc::new(DenseMultilinearExtension::from_evaluations_vec(nv, vals))
    }

    /// Construct the polynomial Q = ∏(y2 - y1_xx) over all x in the domain.
    fn construct_sumcheck_poly(
        &self,
        y1_00: &Rc<DenseMultilinearExtension<F>>,
        y1_01: &Rc<DenseMultilinearExtension<F>>,
        y1_10: &Rc<DenseMultilinearExtension<F>>,
        y1_11: &Rc<DenseMultilinearExtension<F>>,
    ) -> ListOfProductsOfPolynomials<F> {
        let nv = self.num_vars_y2;
        let diff_00 = Self::diff_mle(&self.y2, y1_00);
        let diff_01 = Self::diff_mle(&self.y2, y1_01);
        let diff_10 = Self::diff_mle(&self.y2, y1_10);
        let diff_11 = Self::diff_mle(&self.y2, y1_11);

        let mut poly = ListOfProductsOfPolynomials::new(nv);
        poly.add_product(
            vec![diff_00, diff_01, diff_10, diff_11].into_iter(),
            F::one(),
        );
        poly
    }

    /// Generate a sumcheck proof for Q = 0.
    pub fn prove_sumcheck(&self, rng: &mut impl Rng) -> (SumcheckProof<F>, F, PolynomialInfo) {
        let (y1_00, y1_01, y1_10, y1_11) = self.evaluate_y1_slices();
        let poly = self.construct_sumcheck_poly(&y1_00, &y1_01, &y1_10, &y1_11);
        let asserted_sum = F::zero();
        let proof = MLSumcheck::prove(&poly).expect("Sumcheck proof generation failed");
        let poly_info = poly.info();
        (proof, asserted_sum, poly_info)
    }

    /// Generate a logup proof for inequalities y2 >= max(y1_xx).
    // pub fn prove_inequalities(
    //     &self,
    // ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
    //     type E = crate::E;

    //     let (y1_00, y1_01, y1_10, y1_11) = self.evaluate_y1_slices();

    //     let t: Vec<F> = (0..(1 << self.num_vars_y2))
    //         .map(|i| {
    //             y1_00.evaluations[i]
    //                 .max(y1_01.evaluations[i])
    //                 .max(y1_10.evaluations[i])
    //                 .max(y1_11.evaluations[i])
    //         })
    //         .collect();

    //     let a: Vec<F> = self.y2.evaluations.clone();

    //     let mut transcript = Transcript::new(b"Logup");
    //     let ((pk, ck), commit) = Logup::process::<E>(self.num_vars_y2 as usize, &a);
    //     let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

    //     (commit, proof, a, t)
    // }

    // pub fn prove_inequalities(
    //     &self,
    // ) -> (
    //     Vec<<E as Pairing>::G1Affine>, // Commitments
    //     LogupProof<E>,                 // Logup proof
    //     Vec<F>,                        // Polynomial evaluations (a)
    //     Vec<F>,                        // Target range (t)
    // ) {
    //     type E = crate::E;

    //     // Step 1: Expand y2 to match the dimensions of y1
    //     let num_vars_y1 = self.num_vars_y1;
    //     let num_vars_y2 = self.num_vars_y2;

    //     let mut expanded_y2 = vec![F::zero(); 1 << num_vars_y1];
    //     for i in 0..(1 << num_vars_y2) {
    //         for b1b2 in 0..(1 << (num_vars_y1 - num_vars_y2)) {
    //             let index = b1b2 * (1 << num_vars_y2) + i;
    //             expanded_y2[index] = self.y2.evaluations[i];
    //         }
    //     }

    //     // Step 2: Combine slices of y1 into a single polynomial
    //     let (y1_00, y1_01, y1_10, y1_11) = self.evaluate_y1_slices();
    //     let combined_y1: Vec<F> = (0..(1 << num_vars_y1))
    //         .map(|i| {
    //             let b1b2_index = i >> num_vars_y2;
    //             match b1b2_index {
    //                 0b00 => y1_00.evaluations[i & ((1 << num_vars_y2) - 1)],
    //                 0b01 => y1_01.evaluations[i & ((1 << num_vars_y2) - 1)],
    //                 0b10 => y1_10.evaluations[i & ((1 << num_vars_y2) - 1)],
    //                 0b11 => y1_11.evaluations[i & ((1 << num_vars_y2) - 1)],
    //                 _ => {
    //                     eprintln!(
    //                         "Unexpected b1b2_index: {} for i: {}, num_vars_y1: {}, num_vars_y2: {}",
    //                         b1b2_index, i, num_vars_y1, num_vars_y2
    //                     );
    //                     unreachable!()
    //                 }
    //             }
    //         })
    //         .collect();

    //     // Step 3: Compute a as a[i] = expanded_y2[i] - combined_y1[i]
    //     let a: Vec<F> = expanded_y2
    //         .iter()
    //         .zip(combined_y1.iter())
    //         .map(|(y2_val, y1_val)| *y2_val - *y1_val)
    //         .collect();

    //     // Step 4: Define the target range t as [0, F::MAX]
    //     let range: Vec<F> = (0..=MAX_VALUE_IN_Y).map(|val| F::from(val)).collect();
    //     let ((pk, ck), commit) = Logup::process::<E>(num_vars_y1, &a);

    //     // Step 5: Use Logup to prove that a ∈ t
    //     let mut transcript = Transcript::new(b"Logup");
    //     let proof = Logup::prove::<E>(&a, &range, &pk, &mut transcript);

    //     // Return commitments, proof, and polynomial evaluations
    //     (commit, proof, a, range)
    // }

    // fn expand_y1_with_new_variable(y1: Vec<F>, num_vars: usize) -> Vec<F> {
    //     let new_len = 1 << (num_vars + 1); // 2^(num_vars + 1)
    //     let mut expanded_y1 = vec![F::zero(); new_len];

    //     for i in 0..(1 << num_vars) {
    //         // Copy existing value for both x_4 = 0 and x_4 = 1
    //         expanded_y1[i * 2] = y1[i]; // Corresponds to x_4 = 0
    //         expanded_y1[i * 2 + 1] = y1[i]; // Corresponds to x_4 = 1
    //     }

    //     expanded_y1
    // }

    // Process inequalities
    pub fn process_inequalities(
        &self,
    ) -> (
        Vec<F>,                        // Expanded y2
        Vec<F>,                        // Combined y1
        Vec<F>,                        // Polynomial evaluations (a)
        Vec<F>,                        // Target range (t)
        Vec<<E as Pairing>::G1Affine>, // Commitments
        MultilinearProverParam<E>,     // Public keys (pk)
        MultilinearVerifierParam<E>,   // Public keys (ck)
    ) {
        type E = crate::E;

        // Step 1: Expand y2 to match the dimensions of y1
        let num_vars_y1 = self.num_vars_y1;
        let num_vars_y2 = self.num_vars_y2;

        let mut expanded_y2 = vec![F::zero(); 1 << num_vars_y1];
        for i in 0..(1 << num_vars_y2) {
            for b1b2 in 0..(1 << (num_vars_y1 - num_vars_y2)) {
                let index = b1b2 * (1 << num_vars_y2) + i;
                expanded_y2[index] = self.y2.evaluations[i];
            }
        }

        // Step 2: Combine slices of y1 into a single polynomial
        let (y1_00, y1_01, y1_10, y1_11) = self.evaluate_y1_slices();
        let combined_y1: Vec<F> = (0..(1 << num_vars_y1))
            .map(|i| {
                let b1b2_index = i >> num_vars_y2;
                match b1b2_index {
                    0b00 => y1_00.evaluations[i & ((1 << num_vars_y2) - 1)],
                    0b01 => y1_01.evaluations[i & ((1 << num_vars_y2) - 1)],
                    0b10 => y1_10.evaluations[i & ((1 << num_vars_y2) - 1)],
                    0b11 => y1_11.evaluations[i & ((1 << num_vars_y2) - 1)],
                    _ => unreachable!(),
                }
            })
            .collect();

        // Step 3: Compute a as a[i] = expanded_y2[i] - combined_y1[i]
        let a: Vec<F> = expanded_y2
            .iter()
            .zip(combined_y1.iter())
            .map(|(y2_val, y1_val)| *y2_val - *y1_val)
            .collect();

        // Step 4: Define the target range t as [0, F::MAX]
        let range: Vec<F> = (0..=MAX_VALUE_IN_Y).map(|val| F::from(val)).collect();

        // println!("a.len() = {}", a.len());

        // Step 5: Generate public keys and commitments
        // let ((pk, ck), commit) = Logup::process::<E>(num_vars_y1, &a);
        let ((pk, ck), commit) = Logup::process::<E>(18, &a);

        (expanded_y2, combined_y1, a, range, commit, pk, ck)
    }

    // Prove inequalities
    pub fn prove_inequalities(
        &self,
        a: &Vec<F>,
        range: &Vec<F>,
        pk: &MultilinearProverParam<E>,
        commit: Vec<<E as Pairing>::G1Affine>,
    ) -> (
        Vec<<E as Pairing>::G1Affine>, // Commitments
        LogupProof<E>,                 // Logup proof
        Vec<F>,                        // Polynomial evaluations (a)
        Vec<F>,                        // Target range (t)
    ) {
        type E = crate::E;

        // Step 6: Use Logup to prove that a ∈ t
        let mut transcript = Transcript::new(b"Logup");
        // let ((_, _), commit) = Logup::process::<E>(self.num_vars_y1, a);
        let proof = Logup::prove::<E>(a, range, pk, &mut transcript);

        (commit, proof, a.clone(), range.clone())
    }
}

//test
#[cfg(test)]
mod tests {

    use super::*;
    use ark_poly::evaluations;
    use ark_std::test_rng;

    #[test]
    fn test_reorder_variable_groups() {
        let num_vars = 4;
        let evaluations: Vec<F> = vec![
            F::from(0u64),
            F::from(1u64),
            F::from(2u64),
            F::from(3u64),
            F::from(4u64),
            F::from(5u64),
            F::from(6u64),
            F::from(7u64),
            F::from(8u64),
            F::from(9u64),
            F::from(10u64),
            F::from(11u64),
            F::from(12u64),
            F::from(0u64),
            F::from(1u64),
            F::from(2u64),
        ];

        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        let group_sizes = vec![2, 1, 1];
        let new_order = vec![1, 2, 0];

        let reordered_poly = reorder_variable_groups(&poly, &group_sizes, &new_order);

        let expected_evaluations: Vec<F> = vec![
            F::from(0u64),
            F::from(4u64),
            F::from(8u64),
            F::from(12u64),
            F::from(1u64),
            F::from(5u64),
            F::from(9u64),
            F::from(0u64),
            F::from(2u64),
            F::from(6u64),
            F::from(10u64),
            F::from(1u64),
            F::from(3u64),
            F::from(7u64),
            F::from(11u64),
            F::from(2u64),
        ];

        assert_eq!(reordered_poly.evaluations, expected_evaluations);

        println!("Transformation successful, evaluations match expected results!");
    }

    #[test]
    fn test_reorder_variable_groups2() {
        let num_vars = 4;
        let evaluations: Vec<F> = vec![
            F::from(0u64),  // 0000 -> 0000
            F::from(1u64),  // 0001 -> 0100
            F::from(2u64),  // 0010 -> 1000
            F::from(3u64),  // 0011 -> 1100
            F::from(8u64),  // 0100 -> 0010
            F::from(9u64),  // 0101 -> 0110
            F::from(10u64), // 0110 -> 1010
            F::from(11u64),
            F::from(4u64),
            F::from(5u64),
            F::from(6u64),
            F::from(7u64),
            F::from(12u64),
            F::from(0u64),
            F::from(1u64),
            F::from(2u64),
        ];

        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);
        //从右往左列group size
        let group_sizes = vec![2, 1, 1];
        //从右到左列group的顺序
        let new_order = vec![2, 1, 0];

        let reordered_poly = reorder_variable_groups(&poly, &group_sizes, &new_order);

        let expected_evaluations: Vec<F> = vec![
            F::from(0u64),  // 0000
            F::from(4u64),  // 0001
            F::from(8u64),  // 0010
            F::from(12u64), // 0011
            F::from(1u64),  // 0100
            F::from(5u64),  // 0101
            F::from(9u64),  // 0110
            F::from(0u64),  // 0111
            F::from(2u64),  // 1000
            F::from(6u64),  // 1001
            F::from(10u64), // 1010
            F::from(1u64),  // 1011
            F::from(3u64),  // 1100
            F::from(7u64),  // 1101
            F::from(11u64), // 1110
            F::from(2u64),  // 1111
        ];

        assert_eq!(reordered_poly.evaluations, expected_evaluations);

        println!("Transformation successful, evaluations match expected results!");
    }

    #[test]
    fn test_reorder_variable_groups3() {
        let num_vars = 4;
        let evaluations: Vec<F> = vec![
            F::from(0u64),  // 0000 -> 0000
            F::from(1u64),  // 0001 -> 0100
            F::from(2u64),  // 0010 -> 1000
            F::from(3u64),  // 0011 -> 1100
            F::from(8u64),  // 0100 -> 0010
            F::from(9u64),  // 0101 -> 0110
            F::from(10u64), // 0110 -> 1010
            F::from(11u64), // 0111 -> 1110
            F::from(4u64),  // 1000 -> 0001
            F::from(5u64),  // 1001 -> 0101
            F::from(6u64),  // 1010 -> 1001
            F::from(7u64),  // 1011 -> 1101
            F::from(12u64), // 1100 -> 0011
            F::from(0u64),  // 1101 -> 0111
            F::from(1u64),  // 1110 -> 1011
            F::from(2u64),  // 1111 -> 1111
        ];

        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations);

        let group_sizes = vec![1, 1, 1, 1];
        let new_order = vec![3, 2, 0, 1];

        let reordered_poly = reorder_variable_groups(&poly, &group_sizes, &new_order);

        let expected_evaluations: Vec<F> = vec![
            F::from(0u64),  // 0000
            F::from(4u64),  // 0001
            F::from(8u64),  // 0010
            F::from(12u64), // 0011
            F::from(1u64),  // 0100
            F::from(5u64),  // 0101
            F::from(9u64),  // 0110
            F::from(0u64),  // 0111
            F::from(2u64),  // 1000
            F::from(6u64),  // 1001
            F::from(10u64), // 1010
            F::from(1u64),  // 1011
            F::from(3u64),  // 1100
            F::from(7u64),  // 1101
            F::from(11u64), // 1110
            F::from(2u64),  // 1111
        ];

        assert_eq!(reordered_poly.evaluations, expected_evaluations);

        println!("Transformation successful, evaluations match expected results!");
    }

    #[test]
    fn test_reorder_variable_groups4() {
        let num_vars = 3;
        let evaluations: Vec<F> = vec![
            F::from(0u64), // 000 -> 000
            F::from(1u64), // 001 -> 010
            F::from(2u64), // 010 -> 100
            F::from(3u64), // 011 -> 110
            F::from(4u64), // 100 -> 001
            F::from(5u64), // 101 -> 011
            F::from(6u64), // 110 -> 101
            F::from(7u64), // 111 -> 111
        ];
        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations.clone());

        let group_sizes = vec![2, 1];
        let new_order = vec![1, 0];
        let reordered_poly = reorder_variable_groups(&poly, &group_sizes, &new_order);

        let expected_evaluations: Vec<F> = vec![
            F::from(0u64),
            F::from(4u64),
            F::from(1u64),
            F::from(5u64),
            F::from(2u64),
            F::from(6u64),
            F::from(3u64),
            F::from(7u64),
        ];
        for i in 0..2usize.pow(2) {
            if &evaluations[i] != &reordered_poly.evaluations[i * 2] {
                println!(
                    "i: {}, evaluations[i]: {}, reordered_poly.evaluations[i * 2]: {}",
                    i,
                    evaluations[i],
                    reordered_poly.evaluations[i * 2]
                );
            }
        }
        assert!(reordered_poly.evaluations == expected_evaluations);
    }

    #[test]
    fn test_reorder_variable_groups5() {
        let num_vars = 14;
        let evaluations: Vec<F> = (0..1 << num_vars).map(|i| F::from(i as u64)).collect();

        let poly = DenseMultilinearExtension::from_evaluations_vec(num_vars, evaluations.clone());

        let group_sizes = vec![8, 6];
        let new_order = vec![1, 0];
        let reordered_poly = reorder_variable_groups(&poly, &group_sizes, &new_order);

        // print 2^8
        for i in 0..2usize.pow(8) {
            if &evaluations[i] != &reordered_poly.evaluations[i * 2usize.pow(6)] {
                println!(
                    "i: {}, evaluations[i]: {}, reordered_poly.evaluations[i]: {}",
                    i,
                    evaluations[i],
                    reordered_poly.evaluations[i * 2usize.pow(6)]
                );
            }
        }
    }
}
