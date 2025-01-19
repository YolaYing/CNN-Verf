//! Proving a ReLU Layer using the Logup Protocol
//! Input: Input vector `y1` (scalars) and output vector `y3` (scalars).
//!
//! The protocol proves `y3 = relu(y1 >> 2^Q)`.
//!
//! **Proof Process:**
//! 1. Prover calculates query set `{(y1_i, y3_i)}`.
//! 2. Prover calculates table set `{(x, relu(x >> 2^Q))}`, where `x` is in `[-MAX_VAL_Y1, MAX_VAL_Y1]`.
//! 3. Prover conducts Logup protocol to prove `(y1, y3) ∈ table set`.
//!     - Prover receives a random number `alpha` from the verifier.
//!     - Prover computes `a = y1 + alpha * y3`.
//!     - Prover computes table `t = x + alpha * relu(x >> 2^Q)`.
//!     - Prover proves `a ∈ t` using the Logup protocol.
//! 4. Prover sends the proof to the verifier.
//! 5. Verifier verifies the proof.

use crate::{E, F};
use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField, Zero};
use ark_std::vec::Vec;
use logup::{Logup, LogupProof};
use merlin::Transcript;
use pcs::multilinear_kzg::data_structures::{MultilinearProverParam, MultilinearVerifierParam};

/// Maximum value for `y1`.
const MAX_VAL_Y1: i64 = 131072; //2^17

pub struct Prover {
    pub Q: u64,
    pub y1: Vec<F>,
    pub y3: Vec<F>,
}

impl Prover {
    /// Create a new Prover instance with `y1` and `y3`.
    pub fn new(Q: u64, y1: Vec<F>, y3: Vec<F>) -> Self {
        Prover { Q, y1, y3 }
    }

    /// Compute the table set for the Logup protocol.
    pub fn compute_table_set(&self, alpha: F) -> Vec<F> {
        let mut table = Vec::new();
        let shift_factor = F::from(2u64).pow(&[self.Q]);

        // Attention: Ensure t contains a power-of-2 number of unique values
        for x in -MAX_VAL_Y1..=MAX_VAL_Y1 - 1 {
            let x_field = F::from(x as i64);
            // let shifted_val = F::from((x >> self.Q) as u64);
            let shifted_val = F::from(
                (x_field.into_bigint().as_ref()[0] as i64
                    / shift_factor.into_bigint().as_ref()[0] as i64) as i64,
            );

            let relu_shifted_val = if shifted_val <= F::from(MAX_VAL_Y1) {
                shifted_val
            } else {
                F::from(0u64)
            };
            table.push(x_field + alpha * relu_shifted_val);
        }
        table.sort();
        table.dedup();

        if table.len() != 2 * MAX_VAL_Y1 as usize {
            panic!("Table set does not contain 2 * MAX_VAL_Y1 unique values");
        }

        table
    }

    pub fn compute_a(&self, alpha: F) -> Vec<F> {
        let mut a: Vec<F> = self
            .y1
            .iter()
            .zip(self.y3.iter())
            .map(|(&y1_i, &y3_i)| y1_i + alpha * y3_i)
            .collect();
        // attention a length should be power of 2
        let len = a.len();
        let next_power_of_two = len.next_power_of_two();

        if next_power_of_two > len {
            a.resize(next_power_of_two, F::zero());
        }

        a
    }

    pub fn process_logup(
        &self,
        a: &Vec<F>,
    ) -> (
        Vec<<E as Pairing>::G1Affine>,
        MultilinearProverParam<E>,
        MultilinearVerifierParam<E>,
        // Transcript,
    ) {
        let mut transcript = Transcript::new(b"Logup");
        let ((pk, ck), commit) = Logup::process::<E>(20, &a);

        (commit, pk, ck)
    }

    /// Prove the Logup protocol for `y1` and `y3`.
    pub fn prove_logup(
        &self,
        commit: Vec<<E as Pairing>::G1Affine>,
        pk: MultilinearProverParam<E>,
        a: Vec<F>,
        t: Vec<F>,
        // transcript: &mut Transcript,
    ) -> (Vec<<E as Pairing>::G1Affine>, LogupProof<E>, Vec<F>, Vec<F>) {
        let mut transcript = Transcript::new(b"Logup");
        let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);

        (commit, proof, a, t)
    }
}
