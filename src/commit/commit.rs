// commit.rs
use crate::{E, F};
use ark_std::{rand::Rng, vec::Vec};
use pcs::hyrax_kzg::hyrax_kzg_1::HyraxKzgPCS1;
use pcs::multilinear_kzg::data_structures::{
    MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam,
};
use pcs::PolynomialCommitmentScheme;

// Prepare SRS
pub fn prepare_srs<R: Rng>(rng: &mut R, eval: &Vec<F>) -> MultilinearUniversalParams<E> {
    let num_vars = eval.len().ilog2();
    HyraxKzgPCS1::<E>::gen_srs(rng, num_vars as usize)
}

// Preprocess weights by committing to them
pub fn preprocess_w(
    pp: &MultilinearProverParam<E>,
    w: &Vec<F>,
) -> <HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Commitment {
    HyraxKzgPCS1::commit(pp, w)
}

// Open the commitment to weights
pub fn open_w(
    pp: &MultilinearProverParam<E>,
    w: &Vec<F>,
    point: &Vec<F>,
) -> (<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Proof, F) {
    HyraxKzgPCS1::open(pp, w, point)
}

// Verify the commitment and opening proof of weights
pub fn verify_w(
    vp: &MultilinearVerifierParam<E>,
    commit: &<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Commitment,
    point: &Vec<F>,
    proof: &<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Proof,
    value: F,
) -> bool {
    HyraxKzgPCS1::verify(vp, commit, point, proof, value)
}

// Commit to multiple inputs in convolution and maxpool
pub fn commit_all(
    pp: &MultilinearProverParam<E>,
    inputs: &[Vec<F>],
) -> Vec<<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Commitment> {
    inputs.iter().map(|input| preprocess_w(pp, input)).collect()
}

// Open commitments to all inputs
pub fn open_all(
    pp: &MultilinearProverParam<E>,
    inputs: &[Vec<F>],
    points: &[Vec<F>],
) -> Vec<(<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Proof, F)> {
    inputs
        .iter()
        .zip(points.iter())
        .map(|(input, point)| open_w(pp, input, point))
        .collect()
}

// Verify multiple commitments and their openings
pub fn verify_all(
    vp: &MultilinearVerifierParam<E>,
    commits: &[<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Commitment],
    points: &[Vec<F>],
    proofs: &[<HyraxKzgPCS1<E> as PolynomialCommitmentScheme<E>>::Proof],
    values: &[F],
) -> bool {
    commits
        .iter()
        .zip(points.iter())
        .zip(proofs.iter())
        .zip(values.iter())
        .all(|(((commit, point), proof), value)| verify_w(vp, commit, point, proof, *value))
}

// Helper function to derive Prover and Verifier parameters
pub fn generate_pp(
    srs: &MultilinearUniversalParams<E>,
) -> (MultilinearProverParam<E>, MultilinearVerifierParam<E>) {
    HyraxKzgPCS1::trim(srs)
}
