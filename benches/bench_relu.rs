use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, Criterion};
use merlin::Transcript;
use std::time::Duration;
use zkconv::relu::{prover::Prover, verifier::Verifier};
use zkconv::{E, F};

/// Generate mock data for testing the relu layer proof setup.
/// See detailed steps in the code comments.
fn generate_mock_data(Q: u32, length: usize) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let mut rng = test_rng();

    // Generate y1 as random integers mapped to the field
    let y1_ints: Vec<u64> = (0..length)
        .map(|_| rng.next_u64() % (1 << (Q + 10)))
        .collect();
    let y1: Vec<F> = y1_ints.iter().map(|&x| F::from(x)).collect();

    // Compute y2 = floor(y1 / 2^Q)
    let two_pow_q = 1u64 << Q;
    let y2_ints: Vec<u64> = y1_ints.iter().map(|&x| x >> Q).collect();
    let y2: Vec<F> = y2_ints.iter().map(|&x| F::from(x)).collect();

    // Compute remainder = y1 - 2^Q * y2
    let remainder_ints: Vec<u64> = y1_ints
        .iter()
        .zip(y2_ints.iter())
        .map(|(&y1_i, &y2_i)| y1_i - (y2_i << Q))
        .collect();
    let remainder: Vec<F> = remainder_ints.iter().map(|&x| F::from(x)).collect();

    // Compute y3 = relu(y2)
    let y3 = y2.clone();

    (y1, y2, y3, remainder)
}

fn benchmark_relu_prover_verifier(c: &mut Criterion) {
    let Q: u32 = 4;
    let length = 16;
    let (y1, _y2, _y3, _remainder) = generate_mock_data(Q, length);

    let prover = Prover::new(Q, y1.clone());
    let verifier = Verifier::new(
        Q,
        y1.clone(),
        prover.y2.clone(),
        prover.y3.clone(),
        prover.remainder.clone(),
    );

    // c.bench_function("Prover total prove", |b| {
    //     b.iter(|| {
    //         let mut rng = test_rng();
    //         prover.prove_step1_sumcheck(&mut rng);
    //         prover.prove_step1_logup();
    //         prover.prove_step2_logup();
    //     });
    // });

    // c.bench_function("Verifier total verification", |b| {
    //     let mut rng = test_rng();
    //     let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
    //     let (commit_step1, proof_step1, a_step1, t_step1) = prover.prove_step1_logup();
    //     let (commit_step2, proof_step2, a_step2, t_step2) = prover.prove_step2_logup();

    //     b.iter(|| {
    //         verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
    //         verifier.verify_step1_logup(&commit_step1, &proof_step1, &a_step1, &t_step1);
    //         verifier.verify_step2_logup(&commit_step2, &proof_step2, &a_step2, &t_step2);
    //     });
    // });
    // Precompute logup data outside of the benchmark
    let (commit_step1, pk_step1, ck_step1, t_step1) =
        prover.process_step1_logup(&prover.remainder, prover.Q as usize);
    let (a_step2, t_step2) = prover.compute_a_t(&prover.y2, &prover.y3);
    let mut transcript = Transcript::new(b"Logup");
    let (commit_step2, pk_step2, ck_step2) = prover.process_step2_logup(&prover.y3);

    c.bench_function("Prover total prove", |b| {
        b.iter(|| {
            let mut rng = test_rng();
            prover.prove_step1_sumcheck(&mut rng);
            prover.prove_step1_logup(commit_step1.clone(), pk_step1.clone(), t_step1.clone());
            prover.prove_step2_logup(
                commit_step2.clone(),
                pk_step2.clone(),
                t_step2.clone(),
                a_step2.clone(),
                &mut transcript,
            );
        });
    });

    let mut rng = test_rng();
    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
    let (commit_step1, proof_step1, a_step1, t_step1) =
        prover.prove_step1_logup(commit_step1.clone(), pk_step1.clone(), t_step1.clone());
    let (commit_step2, proof_step2, a_step2, t_step2) = prover.prove_step2_logup(
        commit_step2.clone(),
        pk_step2.clone(),
        t_step2.clone(),
        a_step2.clone(),
        &mut transcript,
    );
    let mut transcript = Transcript::new(b"Logup");

    c.bench_function("Verifier total verification", |b| {
        b.iter(|| {
            verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
            verifier.verify_step1_logup(&commit_step1, &proof_step1, &a_step1, &t_step1, &ck_step1);
            verifier.verify_step2_logup(
                &commit_step2,
                &proof_step2,
                &a_step2,
                &t_step2,
                &ck_step2,
                // &mut transcript,
            );
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_relu_prover_verifier
}
criterion_main!(benches);
