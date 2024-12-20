use crate::{E, F};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_std::vec::Vec;
use logup::{Logup, LogupProof};
use merlin::Transcript;

// For sumcheck
use ark_sumcheck::ml_sumcheck::{
    data_structures::PolynomialInfo, MLSumcheck, Proof as SumcheckProof,
};

pub struct Verifier {
    pub Q: u32,
    pub y1: Vec<F>,
    pub y2: Vec<F>,
    pub y3: Vec<F>,
    pub remainder: Vec<F>,
}

impl Verifier {
    pub fn new(Q: u32, y1: Vec<F>, y2: Vec<F>, y3: Vec<F>, remainder: Vec<F>) -> Self {
        Verifier {
            Q,
            y1,
            y2,
            y3,
            remainder,
        }
    }

    // Verify step1 sumcheck proof
    pub fn verify_step1_sumcheck(
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

    // Verify step1 logup proof: remainder in [0, 2^Q-1]
    // Given commit, proof, a, t from Prover
    pub fn verify_step1_logup(
        &self,
        commit: &Vec<<E as Pairing>::G1Affine>,
        proof: &LogupProof<E>,
        a: &Vec<F>,
        t: &Vec<F>,
    ) -> bool {
        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(20, &a);
        Logup::verify::<E>(a, t, &commit, &ck, proof, &mut transcript)
    }
    // Logup::verify(&a, &t, &commit, &ck, &proof, &mut transcript);

    // Verify step2 logup proof: y3 = relu(y2)
    // Given commit, proof, a, t from Prover
    pub fn verify_step2_logup(
        &self,
        commit: &Vec<<E as Pairing>::G1Affine>,
        proof: &LogupProof<E>,
        a: &Vec<F>,
        t: &Vec<F>,
    ) -> bool {
        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(20, &a);
        Logup::verify::<E>(a, t, &commit, &ck, proof, &mut transcript)
    }
}
