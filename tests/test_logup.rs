use ark_ec::pairing::Pairing;
use ark_std::{rand::seq::SliceRandom, test_rng};
use logup::Logup;
use merlin::Transcript;

type E = ark_bn254::Bn254;
type F = <E as Pairing>::ScalarField;

const M: usize = 1 << 20;
const N: usize = 1 << 8;

#[test]
fn test_logup() {
    let mut rng = test_rng();
    let t: Vec<_> = (1..=N).into_iter().map(|x| F::from(x as u32)).collect();
    let a: Vec<_> = (1..=M)
        .into_iter()
        .map(|_| t.choose(&mut rng).unwrap().clone())
        .collect();
    let mut transcript = Transcript::new(b"Logup");
    let ((pk, ck), commit) = Logup::process::<E>(20, &a); // srs for 20 variates is enough
    let proof = Logup::prove::<E>(&a, &t, &pk, &mut transcript);
    let mut transcript = Transcript::new(b"Logup");
    Logup::verify(&a, &t, &commit, &ck, &proof, &mut transcript);
}
