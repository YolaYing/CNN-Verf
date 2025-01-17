//! This module implements the verifier side of the proving process.
//!
//! ### Verifier Responsibilities:
//! 1. Selects `r1` and `r`, and has knowledge of the dimensions of `Y`, `W`, and `X`.
//! 2. Sends `r1` to the Prover.
//! 3. Receives proofs for the three sumcheck steps.
//! 4. Verifies the correctness of the received proofs.

use crate::F;
use ark_serialize::CanonicalDeserialize;
use ark_sumcheck::ml_sumcheck::{
    data_structures::PolynomialInfo, MLSumcheck, Proof as SumcheckProof,
};
#[derive(Clone, CanonicalDeserialize)]
pub struct VerifierMessage<F: ark_ff::Field> {
    pub r1_values: Vec<F>,
    pub r: F,
}

pub struct Verifier {
    pub num_vars_j: usize,
    pub num_vars_s: usize,
    pub num_vars_i: usize,
    pub num_vars_a: usize,
    pub num_vars_b: usize,
}

impl Verifier {
    pub fn new(
        num_vars_j: usize,
        num_vars_s: usize,
        num_vars_i: usize,
        num_vars_a: usize,
        num_vars_b: usize,
    ) -> Self {
        Self {
            num_vars_j,
            num_vars_s,
            num_vars_i,
            num_vars_a,
            num_vars_b,
        }
    }

    fn verify_sumcheck_proof(
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

    pub fn verify(
        &self,
        proof_s: &SumcheckProof<F>,
        proof_f: &SumcheckProof<F>,
        proof_g: &SumcheckProof<F>,
        asserted_s: F,
        asserted_f: F,
        asserted_g: F,
        poly_info_s: &PolynomialInfo,
        poly_info_f: &PolynomialInfo,
        poly_info_g: &PolynomialInfo,
    ) -> bool {
        // sum_s Y(r1,s)*R[s]
        if !self.verify_sumcheck_proof(proof_s, asserted_s, poly_info_s) {
            println!("Failed to verify sumcheck proof for sum_s Y(r1,s)*R[s]");
            return false;
        }
        // sum_i sum_a X(i,a)*R[a]
        // let nv_f = self.num_vars_i + self.num_vars_a;
        if !self.verify_sumcheck_proof(proof_f, asserted_f, poly_info_f) {
            println!("Failed to verify sumcheck proof for sum_i sum_a X(i,a)*R[a]");
            return false;
        }
        // sum_i sum_b W'(r1,i,b)*R[b]
        // let nv_g = self.num_vars_i + self.num_vars_b;
        if !self.verify_sumcheck_proof(proof_g, asserted_g, poly_info_g) {
            println!("Failed to verify sumcheck proof for sum_i sum_b W'(r1,i,b)*R[b]");
            return false;
        }

        true
    }
}
