use crate::{E, F};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_std::vec::Vec;
use logup::{Logup, LogupProof};
use merlin::Transcript;
use pcs::multilinear_kzg::data_structures::MultilinearVerifierParam;

pub struct Verifier {
    pub Q: u64,
    pub y1: Vec<F>,
    pub y3: Vec<F>,
}

impl Verifier {
    /// Create a new Verifier instance.
    pub fn new(Q: u64, y1: Vec<F>, y3: Vec<F>) -> Self {
        Verifier { Q, y1, y3 }
    }

    /// Verify Logup proof: Verify `y3 = relu(y1 >> 2^Q)`.
    pub fn verify_logup(
        &self,
        commit: &Vec<<E as Pairing>::G1Affine>,
        proof: &LogupProof<E>,
        a: &Vec<F>,
        t: &Vec<F>,
        ck: &MultilinearVerifierParam<E>,
    ) -> bool {
        let mut transcript = Transcript::new(b"Logup");
        Logup::verify::<E>(a, t, commit, ck, proof, &mut transcript)
    }
}
