//! Sumcheck Protocol for multilinear extension

use crate::ml_sumcheck::data_structures::{ListOfProductsOfPolynomials, PolynomialInfo};
use crate::ml_sumcheck::protocol::prover::{ProverMsg, ProverState};
use crate::ml_sumcheck::protocol::verifier::SubClaim;
use crate::ml_sumcheck::protocol::IPForMLSumcheck;
use crate::rng::{Blake2b512Rng, FeedableRNG};
use ark_bn254::G1Affine;
use ark_ff::Field;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::hyrax::HyraxPC;
use ark_poly_commit::PolynomialCommitment;
use ark_std::marker::PhantomData;
use ark_std::vec::Vec;
pub type F = <G1Affine as ark_ec::AffineRepr>::ScalarField;
pub type HyraxPCS = HyraxPC<G1Affine, DenseMultilinearExtension<F>>;

pub mod protocol;

pub mod data_structures;
#[cfg(test)]
mod test;

/// Sumcheck for products of multilinear polynomial
pub struct MLSumcheck<F: Field>(#[doc(hidden)] PhantomData<F>);

/// proof generated by prover
pub type Proof<F> = Vec<ProverMsg<F>>;

impl<F: Field> MLSumcheck<F> {
    /// extract sum from the proof
    pub fn extract_sum(proof: &Proof<F>) -> F {
        proof[0].evaluations[0] + proof[0].evaluations[1]
    }

    /// generate proof of the sum of polynomial over {0,1}^`num_vars`
    ///
    /// The polynomial is represented by a list of products of polynomials along with its coefficient that is meant to be added together.
    ///
    /// This data structure of the polynomial is a list of list of `(coefficient, DenseMultilinearExtension)`.
    /// * Number of products n = `polynomial.products.len()`,
    /// * Number of multiplicands of ith product m_i = `polynomial.products[i].1.len()`,
    /// * Coefficient of ith product c_i = `polynomial.products[i].0`
    ///
    /// The resulting polynomial is
    ///
    /// $$\sum_{i=0}^{n}C_i\cdot\prod_{j=0}^{m_i}P_{ij}$$
    pub fn prove(polynomial: &ListOfProductsOfPolynomials<F>) -> Result<Proof<F>, crate::Error> {
        let mut fs_rng = Blake2b512Rng::setup();
        Self::prove_as_subprotocol(&mut fs_rng, polynomial).map(|r| r.0)
    }

    /// This function does the same thing as `prove`, but it uses a `FeedableRNG` as the transcript/to generate the
    /// verifier challenges. Additionally, it returns the prover's state in addition to the proof.
    /// Both of these allow this sumcheck to be better used as a part of a larger protocol.
    pub fn prove_as_subprotocol(
        fs_rng: &mut impl FeedableRNG<Error = crate::Error>,
        polynomial: &ListOfProductsOfPolynomials<F>,
    ) -> Result<(Proof<F>, ProverState<F>), crate::Error> {
        fs_rng.feed(&polynomial.info())?;

        let pp = HyraxPCS::setup(polynomial.num_variables, None, fs_rng).unwrap();
        let (ck, _) = HyraxPCS::trim(&pp, polynomial.num_variables, 1, None).unwrap();

        let mut prover_state = IPForMLSumcheck::prover_init(polynomial, &ck);
        let mut verifier_msg = None;
        let mut prover_msgs = Vec::with_capacity(polynomial.num_variables);
        for _ in 0..polynomial.num_variables {
            let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, &verifier_msg, &ck);
            fs_rng.feed(&prover_msg)?;
            prover_msgs.push(prover_msg);
            verifier_msg = Some(IPForMLSumcheck::sample_round(fs_rng));
        }
        prover_state
            .randomness
            .push(verifier_msg.unwrap().randomness);

        Ok((prover_msgs, prover_state))
    }

    /// verify the claimed sum using the proof
    pub fn verify(
        polynomial_info: &PolynomialInfo,
        claimed_sum: F,
        proof: &Proof<F>,
    ) -> Result<SubClaim<F>, crate::Error> {
        let mut fs_rng = Blake2b512Rng::setup();
        Self::verify_as_subprotocol(&mut fs_rng, polynomial_info, claimed_sum, proof)
    }

    /// This function does the same thing as `prove`, but it uses a `FeedableRNG` as the transcript/to generate the
    /// verifier challenges. This allows this sumcheck to be used as a part of a larger protocol.
    pub fn verify_as_subprotocol(
        fs_rng: &mut impl FeedableRNG<Error = crate::Error>,
        polynomial_info: &PolynomialInfo,
        claimed_sum: F,
        proof: &Proof<F>,
    ) -> Result<SubClaim<F>, crate::Error> {
        fs_rng.feed(polynomial_info)?;

        let pp = HyraxPCS::setup(polynomial_info.num_variables, None, fs_rng).unwrap();
        let (_, vk) = HyraxPCS::trim(&pp, polynomial_info.num_variables, 1, None).unwrap();

        let mut verifier_state = IPForMLSumcheck::verifier_init(polynomial_info, Some(vk));
        for i in 0..polynomial_info.num_variables {
            let prover_msg = proof.get(i).expect("proof is incomplete");
            fs_rng.feed(prover_msg)?;
            let _verifier_msg = IPForMLSumcheck::verify_round(
                (*prover_msg).clone(),
                &mut verifier_state,
                &vk,
                fs_rng,
            );
        }

        IPForMLSumcheck::check_and_generate_subclaim(verifier_state, claimed_sum)
    }
}
