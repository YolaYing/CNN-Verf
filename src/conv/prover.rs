//! This file implements the proving process for a convolution layer using sumcheck protocol.
//!
//! ### Input:
//! 1. A vector of scalars `Y`, represented as a multi-variate polynomial `Y(j, s)`:
//!    - `j`: Index of the output channel (`j = [j_0, j_1, ..., j_log(c_out)]`).
//!    - `s`: Index of the output position (`s = [s_0, s_1, ..., s_log(n_y)]`).
//! 2. A vector of scalars `W`, represented as a multi-variate polynomial `W(i, j, b)`:
//!    - `i`: Index of the input channel (`i = [i_0, i_1, ..., i_log(c_in)]`).
//!    - `j`: Index of the output channel (`j = [j_0, j_1, ..., j_log(c_out)]`).
//!    - `b`: Index of the kernel position (`b = [b_0, b_1, ..., b_m]`).
//! 3. A vector of scalars `X`, represented as a multi-variate polynomial `X(i, a)`:
//!    - `i`: Index of the input channel (`i = [i_0, i_1, ..., i_log(c_in)]`).
//!    - `a`: Index of the input position (`a = [a_0, a_1, ..., a_n_x]`).
//! 4. Basic information:
//!    - `c_out`: Number of output channels.
//!    - `c_in`: Number of input channels.
//!    - `stride`: Stride of the convolution.
//!    - `padding`: Padding of the convolution.
//!    - `n_x`: Width of the input.
//!    - `n_y`: Width of the output.
//!    - `m`: Width of the kernel.
//!
//! ### Preprocessing:
//! 1. The verifier sends `r` to replace `Z`, and `r1` (a list of `r1_0, r1_1, ..., r1_log(c_out)`) to replace `j` (`j_0, j_1, ..., j_log(c_out)`).
//! 2. Compute `Y(r1, s)` from `Y(j, s)` using `Y.fix_variables(&[r1])` to obtain a new multilinear polynomial `Y(r1, s)`.
//! 3. Compute `W'(j, i, b)` from `W(i, j, b)` using the `reorder_variable_groups` function to get `W'(j, i, b)`.
//! 4. Compute `W'(r1, i, b)` from `W'(j, i, b)` using `W'.fix_variables(&[r1])` to obtain a new multilinear polynomial `W'(r1, i, b)`.
//! 5. Compute `R[s]`, `R[a]`, and `R[b]`:
//!    - `R[a]`'s MLE: `R[a] = R(a1, a2, ..., a_n_x) = (a1 * r + (1 - a1)) * (a2 * r^2 + (1 - a2)) * ...`
//!    - `R[b]`'s MLE: `R[b] = R(b1, b2, ..., b_m) = (b1 * r + (1 - b1)) * (b2 * r^2 + (1 - b2)) * ...`
//!    - `R[s]`'s MLE: `R[s] = R(s1, s2, ..., s_n_y) = (s1 * r + (1 - s1)) * (s2 * r^2 + (1 - s2)) * ...`
//! 6. Compute `overall_sum`: `overall_sum = sum_s Y(r1, s) * R[s]`.
//! 7. Compute `f_sum`: `f_sum = sum_i sum_a X(i, a) * R[a]`.
//! 8. Compute `g_sum`: `g_sum = sum_b W'(r1, i, b) * R[b]`.
//!
//! ### Proving Process:
//! 1. Use the sumcheck protocol to prove `sum_s Y(r1, s) * R[s] = target`:
//!    - Specifically, `sum_{s0, s1, ..., s_logn} Y(r1, s0, s1, ..., s_logn) * R[s0, s1, ..., s_logn] = target`.
//! 2. Use the sumcheck protocol to prove `sum_i f'(i) = f_sum`, where `f'(i) = sum_a X(i, a) * R[a]`.
//! 3. Use the sumcheck protocol to prove `sum_i g'(i) = g_sum`, where `g'(i) = sum_b W'(r1, i, b) * R[b]`.

use super::verifier::VerifierMessage;
use crate::F;
use ark_ff::{Field, One, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::vec::Vec;
use ark_sumcheck::ml_sumcheck::{
    data_structures::{ListOfProductsOfPolynomials, PolynomialInfo},
    MLSumcheck, Proof as SumcheckProof,
};

#[derive(Clone)]
pub struct Prover {
    pub Y: Rc<DenseMultilinearExtension<F>>, // Multilinear polynomial Y(j, s)
    pub W: Rc<DenseMultilinearExtension<F>>, // Multilinear polynomial W(i, j, b)
    pub X: Rc<DenseMultilinearExtension<F>>, // Multilinear polynomial X(i, a)

    pub num_vars_j: usize,
    pub num_vars_s: usize,
    pub num_vars_i: usize,
    pub num_vars_a: usize,
    pub num_vars_b: usize,
}

impl Prover {
    pub fn new(
        Y: Rc<DenseMultilinearExtension<F>>,
        W: Rc<DenseMultilinearExtension<F>>,
        X: Rc<DenseMultilinearExtension<F>>,
        num_vars_j: usize,
        num_vars_s: usize,
        num_vars_i: usize,
        num_vars_a: usize,
        num_vars_b: usize,
    ) -> Self {
        Self {
            Y,
            W,
            X,
            num_vars_j,
            num_vars_s,
            num_vars_i,
            num_vars_a,
            num_vars_b,
        }
    }

    fn fix_Y(&self, r1_values: &[F]) -> Rc<DenseMultilinearExtension<F>> {
        let fixed = self.Y.fix_variables(r1_values);
        Rc::new(fixed)
    }

    fn reorder_variable_groups(
        poly: &DenseMultilinearExtension<F>,
        group_sizes: &[usize],
        new_order: &[usize],
    ) -> DenseMultilinearExtension<F> {
        // Reorder the variables of `poly` according to `new_order` of groups.
        // Steps:
        // 1. Compute original offsets
        let mut original_offsets = Vec::with_capacity(group_sizes.len());
        let mut acc = 0;
        for &size in group_sizes {
            original_offsets.push(acc);
            acc += size;
        }
        let num_vars = poly.num_vars;
        assert_eq!(acc, num_vars, "sum of group_sizes must equal num_vars");

        // Compute new offsets based on new_order
        let mut new_group_offsets = Vec::with_capacity(group_sizes.len());
        let mut cur = 0;
        for &g in new_order {
            new_group_offsets.push(cur);
            cur += group_sizes[g];
        }

        // We now have a permutation of groups. We need a permutation of each variable's position.
        // Create a mapping from old var index to new var index
        let mut var_map = vec![0; num_vars];
        {
            let mut current_new_offset = vec![0; group_sizes.len()];
            for (new_gpos, &old_g) in new_order.iter().enumerate() {
                let start_old = original_offsets[old_g];
                let size_old = group_sizes[old_g];
                let start_new = new_group_offsets[new_gpos];
                for k in 0..size_old {
                    var_map[start_old + k] = start_new + k;
                }
            }
        }

        // Reorder evaluations:
        // For each old_index in [0..2^num_vars], compute new_index by rearranging bits.
        let size = 1 << num_vars;
        let mut new_evals = vec![F::zero(); size];
        for old_index in 0..size {
            let mut new_index = 0;
            for v in 0..num_vars {
                let bit = (old_index >> v) & 1;
                let new_pos = var_map[num_vars - 1 - v];
                new_index |= bit << num_vars - 1 - new_pos;
            }
            new_evals[new_index] = poly.evaluations[old_index];
        }

        DenseMultilinearExtension::from_evaluations_vec(num_vars, new_evals)
    }

    fn fix_W(&self, r1_values: &[F]) -> Rc<DenseMultilinearExtension<F>> {
        // reorder (i,j,b) -> (j,i,b)
        // let W_prime = Self::reorder_variable_groups(
        //     &self.W,
        //     &[self.num_vars_i, self.num_vars_j, self.num_vars_b],
        //     &[1, 0, 2],
        // );
        let fixed = self.W.fix_variables(r1_values);
        Rc::new(fixed)
    }

    fn construct_R_poly(&self, nv: usize, r: F) -> Rc<DenseMultilinearExtension<F>> {
        // R[...] = product over k in [0..nv]: if bit=1 => r^(k) else 1
        // Actually from the description:
        // R[var_bit] = var_bit * r^k+ (1 - var_bit)
        // which simplifies to:
        // if var_bit = 1: value = r^k
        // if var_bit = 0: value = 1
        // final = product of these per variable bits
        let mut evals = Vec::with_capacity(1 << nv);
        for i in 0..(1 << nv) {
            let mut value = F::one();
            for k in 0..nv {
                let bit = (i >> k) & 1;
                value *= if bit == 1 {
                    r.pow(&[(k + 1) as u64])
                } else {
                    F::one()
                };
            }
            evals.push(value);
        }

        Rc::new(DenseMultilinearExtension::from_evaluations_vec(nv, evals))
    }

    fn prove_sumcheck_poly(&self, poly: &ListOfProductsOfPolynomials<F>) -> (SumcheckProof<F>, F) {
        let proof = MLSumcheck::prove(poly).expect("fail to prove sumcheck");
        let asserted_sum = MLSumcheck::extract_sum(&proof);
        (proof, asserted_sum)
    }

    pub fn prove(
        &self,
        rng: &mut impl Rng,
        verifier_msg: VerifierMessage<F>,
    ) -> (
        SumcheckProof<F>,
        SumcheckProof<F>,
        SumcheckProof<F>,
        F,
        F,
        F,
        PolynomialInfo,
        PolynomialInfo,
        PolynomialInfo,
    ) {
        let r1_values = verifier_msg.r1_values;
        let Y_fixed = self.fix_Y(&r1_values);
        let W_fixed = self.fix_W(&r1_values);

        let R_s = self.construct_R_poly(self.num_vars_s, verifier_msg.r);
        let R_a = self.construct_R_poly(self.num_vars_a, verifier_msg.r);
        let R_b = self.construct_R_poly(self.num_vars_b, verifier_msg.r);

        // 1) sum_s Y(r1,s)*R[s]
        let mut poly_s = ListOfProductsOfPolynomials::new(self.num_vars_s);
        poly_s.add_product(vec![Y_fixed.clone(), R_s.clone()].into_iter(), F::one());
        let (proof_s, asserted_s) = self.prove_sumcheck_poly(&poly_s);

        // 2) sum_i sum_a X(i,a)*R[a]
        let poly_f_nv = self.num_vars_i + self.num_vars_a;
        let mut poly_f = ListOfProductsOfPolynomials::new(poly_f_nv);
        // extend R_a to the same number of variables as X
        let factor = 1 << self.num_vars_i; // there are 2^(num_vars_i) points on i
        let original_len = R_a.evaluations.len(); // orginal length is 2^(num_vars_a)
        assert_eq!(original_len, 1 << self.num_vars_a);
        let mut R_a_expanded_evals = Vec::with_capacity(1 << (self.num_vars_i + self.num_vars_a));
        for _ in 0..factor {
            R_a_expanded_evals.extend_from_slice(&R_a.evaluations);
        }
        let R_a_expanded = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            poly_f_nv,
            R_a_expanded_evals,
        ));

        poly_f.add_product(
            vec![self.X.clone(), R_a_expanded.clone()].into_iter(),
            F::one(),
        );
        // poly_f.add_product(vec![self.X.clone(), R_a.clone()].into_iter(), F::one());
        let (proof_f, asserted_f) = self.prove_sumcheck_poly(&poly_f);

        // 3) sum_i sum_b W'(r1,i,b)*R[b]
        let poly_g_nv = self.num_vars_i + self.num_vars_b;
        let mut poly_g = ListOfProductsOfPolynomials::new(poly_g_nv);

        // extend R_b to the same number of variables as W'
        let factor = 1 << self.num_vars_i;
        let original_len = R_b.evaluations.len(); // original length is 2^(num_vars_b)
        assert_eq!(original_len, 1 << self.num_vars_b);
        let mut R_b_expanded_evals = Vec::with_capacity(1 << (self.num_vars_i + self.num_vars_b));
        for _ in 0..factor {
            R_b_expanded_evals.extend_from_slice(&R_b.evaluations);
        }
        let R_b_expanded = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
            poly_g_nv,
            R_b_expanded_evals,
        ));
        poly_g.add_product(
            vec![W_fixed.clone(), R_b_expanded.clone()].into_iter(),
            F::one(),
        );

        // poly_g.add_product(vec![W_fixed.clone(), R_b.clone()].into_iter(), F::one());
        let (proof_g, asserted_g) = self.prove_sumcheck_poly(&poly_g);

        (
            proof_s,
            proof_f,
            proof_g,
            asserted_s,
            asserted_f,
            asserted_g,
            poly_s.info(),
            poly_f.info(),
            poly_g.info(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::Zero;
    use ark_poly::DenseMultilinearExtension;
    use ark_std::rc::Rc;

    #[test]
    fn test_reorder_variable_groups() {
        // Step 1: define a polynomial W(i, j, b) with fixed evaluations
        let num_vars_i = 1; // the dimension of i
        let num_vars_j = 1; // the dimension of j
        let num_vars_b = 1; // the dimension of b
        let group_sizes = vec![num_vars_i, num_vars_j, num_vars_b]; // group sizes

        let total_vars = num_vars_i + num_vars_j + num_vars_b; // total number of variables
        let size = 1 << total_vars; // 2^(1+1+1) = 8

        // fixed evaluations: evaluation = decimal number of binary combination of indices
        let evals: Vec<F> = (0..size).map(|i| F::from(i as u64)).collect();

        let poly = DenseMultilinearExtension::from_evaluations_vec(total_vars, evals.clone());

        println!("Original Polynomial Evaluations:");
        for i in 0..size {
            println!("Index {} -> Value {}", i, evals[i]);
        }

        // Step 2: call reorder_variable_groups to reorder the variable order from (i, j, b) -> (j, i, b)
        let new_order = vec![1, 0, 2]; // new order: (j, i, b)
        let reordered_poly = Prover::reorder_variable_groups(&poly, &group_sizes, &new_order);

        // Step 3: verify the reordered polynomial evaluations
        println!("Reordered Polynomial Evaluations:");
        for i in 0..size {
            println!("Index {} -> Value {}", i, reordered_poly.evaluations[i]);
        }
    }

    #[test]
    fn test_construct_R_poly_function() {
        // Step 1: initialize parameters
        let nv = 2; // number of variables
        let r = F::from(2u64); // r = 2

        // Step 2: call construct_R_poly
        let prover = Prover {
            Y: Rc::new(DenseMultilinearExtension::zero()),
            W: Rc::new(DenseMultilinearExtension::zero()),
            X: Rc::new(DenseMultilinearExtension::zero()),
            num_vars_j: 0,
            num_vars_s: 0,
            num_vars_i: 0,
            num_vars_a: nv,
            num_vars_b: 0,
        };

        let r_poly = prover.construct_R_poly(nv, r);

        // Step 3: calculate the expected values
        let expected_values = vec![
            F::from(1u64), // Index 0
            F::from(2u64), // Index 1
            F::from(4u64), // Index 2
            F::from(8u64), // Index 3
        ];

        // Step 4: verify the computed values
        for (i, expected) in expected_values.iter().enumerate() {
            let computed = r_poly.evaluations[i];
            assert_eq!(
                computed, *expected,
                "Mismatch at index {}: expected {:?}, got {:?}",
                i, expected, computed
            );
            println!(
                "Index {}: Computed Value = {:?}, Expected Value = {:?}",
                i, computed, expected
            );
        }

        println!("test_construct_R_poly_function passed!");
    }
}
