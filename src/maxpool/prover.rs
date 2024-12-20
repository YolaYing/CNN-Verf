// to prove a maxpooling layer:
// input:
//  1. a vector of scalars y1, can be represented as a multi-variate polynomial y1(b1, b2, ..., bn)
//  2. maxpooling result y2, can be represented as a multi-variate polynomial y2(b3,b4, ..., bn)
// the prove process:
//  1. compute y1(0,0,b3,..,bn), y1(0,1,b3,..,bn), y1(1,0,b3,...,bn), y1(1,1,b3,..,bn) from y1
//  2. use sumcheck protocol to prove sum_{b3,b4,...,bn}(y2(b3,b4,...,bn)-y1(0,0,b3,...,bn)(y2(b3,b4,...,bn)-y1(0,1,b3,...,bn)(y2(b3,b4,...,bn)-y1(1,0,b3,...,bn)(y2(b3,b4,...,bn)-y1(1,1,b3,...,bn) = 0
//  3. use logup protocol to prove y2>=y1(0,0,b3,...,bn), y2>=y1(0,1,b3,...,bn), y2>=y1(1,0,b3,...,bn), y2>=y1(1,1,b3,...,bn)
//      f = y2(b3,b4,...,bn) - y1(b1,b2,...,bn)>=0

use crate::{E, F};
use ark_crypto_primitives::crh::sha256::digest::typenum::Length;
// Import F from lib.rs
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

// MAX_VALUE_IN_Y
const MAX_VALUE_IN_Y: u64 = 65536;

pub struct Prover {
    pub y1: Rc<DenseMultilinearExtension<F>>, // Use F instead of Fr
    pub y2: Rc<DenseMultilinearExtension<F>>,
    pub num_vars_y1: usize,
    pub num_vars_y2: usize,
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
    pub fn prove_inequalities(
        &self,
    ) -> (
        Vec<<E as Pairing>::G1Affine>, // Commitments
        LogupProof<E>,                 // Logup proof
        Vec<F>,                        // Polynomial evaluations (a)
        Vec<F>,                        // Target range (t)
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
                    _ => {
                        eprintln!(
                            "Unexpected b1b2_index: {} for i: {}, num_vars_y1: {}, num_vars_y2: {}",
                            b1b2_index, i, num_vars_y1, num_vars_y2
                        );
                        unreachable!()
                    }
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

        // Step 5: Use Logup to prove that a ∈ t
        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(num_vars_y1, &a);
        let proof = Logup::prove::<E>(&a, &range, &pk, &mut transcript);

        // Return commitments, proof, and polynomial evaluations
        (commit, proof, a, range)
    }
}
