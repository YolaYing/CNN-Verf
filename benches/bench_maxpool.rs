use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;
use zkconv::maxpool::{prover::Prover, verifier::Verifier};
use zkconv::F;

fn benchmark_maxpool_prover_verifier(c: &mut Criterion) {
    let mut rng = test_rng();

    let num_vars_y1 = 16; // y1 has 16 variables
    let num_vars_y2 = 14; // y2 has 14 variables after max operation

    let size_y1 = 1 << num_vars_y1; // Total evaluations for y1
    let size_y2 = 1 << num_vars_y2; // Total evaluations for y2

    // Generate random evaluations for y1
    let mut y1_values = Vec::with_capacity(size_y1);
    for _ in 0..size_y1 {
        y1_values.push(F::from(rng.gen_range(0u32..100)));
    }

    // Generate y2 as the maximum of y1 slices for each combination of the reduced variables
    let stride_b1b2 = 1 << (num_vars_y1 - num_vars_y2); // Number of evaluations for each (b3, ..., b16)
    let mut y2_values = Vec::with_capacity(size_y2);

    for sub_i in 0..size_y2 {
        let mut max_val = F::zero();
        for i in 0..stride_b1b2 {
            let index = i * size_y2 + sub_i; // Compute the index for this slice
            max_val = max_val.max(y1_values[index]);
        }
        y2_values.push(max_val);
    }

    let y1 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars_y1,
        y1_values,
    ));
    let y2 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars_y2,
        y2_values,
    ));

    // Prover setup
    let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);
    let (expanded_y2, combined_y1, a, range, commit, pk, ck) = prover.process_inequalities();

    c.bench_function("Maxpool Prover total prove", |b| {
        b.iter(|| {
            prover.prove_sumcheck(&mut rng);
            prover.prove_inequalities(&a, &range, &pk, commit.clone());
        });
    });

    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut rng);
    let (commit_logup, logup_proof, a_proof, range_proof) =
        prover.prove_inequalities(&a, &range, &pk, commit.clone());

    // Verifier setup
    let verifier = Verifier::new(num_vars_y1);

    c.bench_function("Maxpool Verifier total verification", |b| {
        b.iter(|| {
            verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
            verifier.verify_inequalities(&commit_logup, &logup_proof, &a_proof, &range_proof, &ck);
        });
    });
    // let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);

    // c.bench_function("Maxpool Prover total prove", |b| {
    //     b.iter(|| {
    //         prover.prove_sumcheck(&mut rng);
    //         prover.prove_inequalities();
    //     });
    // });

    // let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut rng);
    // let (commit, logup_proof, a, range) = prover.prove_inequalities();

    // // Verifier setup
    // let verifier = Verifier::new(num_vars_y1);

    // c.bench_function("Maxpool Verifier total verification", |b| {
    //     b.iter(|| {
    //         verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
    //         verifier.verify_inequalities(&commit, &logup_proof, &a, &range);
    //     });
    // });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_maxpool_prover_verifier
}
criterion_main!(benches);
