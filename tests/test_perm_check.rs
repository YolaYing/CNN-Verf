use arithmetic::multilinear_poly::evaluate_on_point;
use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;
use ark_std::{test_rng, One, UniformRand};
use merlin::Transcript;
use poly_iop::perm_check::PermCheck;
use rand::seq::SliceRandom;

type F = <Bls12_381 as Pairing>::ScalarField;

#[test]
fn test_perm_check() {
    let mut rng = test_rng();
    let evals_0: Vec<_> = (0..4096).map(|_| F::rand(&mut rng)).collect();
    let mut evals_1 = evals_0.clone();
    evals_1.shuffle(&mut rng);

    let mut transcript = Transcript::new(b"PermCheck");
    let (mut proof, _, _) = PermCheck::prove(evals_0.clone(), evals_1.clone(), &mut transcript);

    let prod_0 = evals_0.iter().fold(F::one(), |mut prod, x| {
        prod *= x;
        prod
    });
    let prod_1 = evals_1.iter().fold(F::one(), |mut prod, x| {
        prod *= x;
        prod
    });
    assert_eq!(prod_0, prod_1);

    let mut transcript = Transcript::new(b"PermCheck");
    let (challenges, values) = PermCheck::verify(12, &mut transcript, &mut proof);
    assert_eq!(values[0], evaluate_on_point(&evals_0, &challenges[0]));
    assert_eq!(values[1], evaluate_on_point(&evals_1, &challenges[1]));
}
