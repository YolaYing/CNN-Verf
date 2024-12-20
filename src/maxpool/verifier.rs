use crate::{E, F}; // Import F and E type aliases from lib.rs
use ark_ec::pairing::Pairing;
use ark_std::vec::Vec;
use ark_sumcheck::ml_sumcheck::{
    data_structures::PolynomialInfo, MLSumcheck, Proof as SumcheckProof,
};
use logup::{Logup, LogupProof};
use merlin::Transcript;

pub struct Verifier {
    pub num_vars_y1: usize, // Number of variables for y1
}

impl Verifier {
    /// Create a new Verifier instance
    pub fn new(num_vars_y1: usize) -> Self {
        Self { num_vars_y1 }
    }

    /// Verify the sumcheck proof for Q = 0
    pub fn verify_sumcheck(
        &self,
        proof: &SumcheckProof<F>,
        asserted_sum: F,
        poly_info: &PolynomialInfo,
    ) -> bool {
        match MLSumcheck::verify(poly_info, asserted_sum, proof) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Verify the logup proof for y2 >= max(y1_xx)
    pub fn verify_inequalities(
        &self,
        commit: &Vec<<E as Pairing>::G1Affine>,
        proof: &LogupProof<E>,
        a: &Vec<F>,
        range: &Vec<F>,
    ) -> bool {
        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(self.num_vars_y1 as usize, &a);
        Logup::verify::<E>(a, range, &commit, &ck, proof, &mut transcript)
    }
}
