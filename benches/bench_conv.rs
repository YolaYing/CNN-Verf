use ark_ff::{Field, UniformRand};
use ark_poly::DenseMultilinearExtension;
use ark_std::rc::Rc;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;
use zkconv::conv::prover::Prover;
use zkconv::conv::verifier::{Verifier, VerifierMessage};
use zkconv::{E, F};

fn read_and_prepare_data() -> (Vec<F>, Vec<F>, Vec<F>) {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open("output_data.txt").expect("Unable to open data file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|line| line.unwrap());

    fn skip_prefix(lines: &mut impl Iterator<Item = String>, prefix: &str) -> String {
        while let Some(line) = lines.next() {
            if line.trim() == prefix {
                return lines
                    .next()
                    .unwrap_or_else(|| panic!("Missing data after prefix: {}", prefix));
            }
        }
        panic!("Prefix not found: {}", prefix);
    }

    fn parse_numbers(line: &str) -> Vec<F> {
        line.split_whitespace()
            .filter_map(|token| token.parse::<F>().ok())
            .collect()
    }

    let x_line = skip_prefix(&mut lines, "X:");
    let w_line = skip_prefix(&mut lines, "W:");
    let y_line = skip_prefix(&mut lines, "Y:");

    let x: Vec<F> = parse_numbers(&x_line);
    let w: Vec<F> = parse_numbers(&w_line);
    let y: Vec<F> = parse_numbers(&y_line);

    (x, w, y)
}

fn benchmark_prover_verifier(c: &mut Criterion) {
    let (x, w, y) = read_and_prepare_data();

    let num_vars_j = 6;
    let num_vars_s = 11;
    let num_vars_i = 2;
    let num_vars_a = 11;
    let num_vars_b = 7;

    let y_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_j + num_vars_s, y);
    let x_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_i + num_vars_a, x);
    let w_poly =
        DenseMultilinearExtension::from_evaluations_vec(num_vars_i + num_vars_j + num_vars_b, w);

    let prover = Prover::new(
        Rc::new(y_poly),
        Rc::new(w_poly),
        Rc::new(x_poly),
        num_vars_j,
        num_vars_s,
        num_vars_i,
        num_vars_a,
        num_vars_b,
    );

    let verifier = Verifier::new(num_vars_j, num_vars_s, num_vars_i, num_vars_a, num_vars_b);

    let mut rng = test_rng();
    let r1_values: Vec<F> = (0..num_vars_j).map(|_| F::rand(&mut rng)).collect();
    let r = F::rand(&mut rng);
    let verifier_msg = VerifierMessage { r1_values, r };

    c.bench_function("Prover prove", |b| {
        b.iter(|| {
            prover.prove(&mut rng, verifier_msg.clone());
        });
    });

    let (
        proof_s,
        proof_f,
        proof_g,
        asserted_s,
        asserted_f,
        asserted_g,
        poly_info_s,
        poly_info_f,
        poly_info_g,
    ) = prover.prove(&mut rng, verifier_msg.clone());

    c.bench_function("Verifier verify", |b| {
        b.iter(|| {
            verifier.verify(
                &proof_s,
                &proof_f,
                &proof_g,
                asserted_s.clone(),
                asserted_f.clone(),
                asserted_g.clone(),
                &poly_info_s,
                &poly_info_f,
                &poly_info_g,
            )
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_prover_verifier
}
criterion_main!(benches);
