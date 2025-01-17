use ark_std::test_rng;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
};
use criterion::{criterion_group, criterion_main, Criterion};
use merlin::Transcript;
use std::fs;
use std::path::Path;
use std::time::Duration;
use zkconv::relu_old::{prover::Prover, verifier::Verifier};
use zkconv::F;

fn read_relu_data<P: AsRef<Path>>(
    file_path: P,
) -> io::Result<(Vec<F>, Vec<F>, Vec<F>, Vec<F>, usize, usize, usize)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Parse input header
    let input_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing input header"))??;
    let input_dims: Vec<usize> = input_header
        .split_whitespace()
        .skip(2) // Skip "relu input(y1)"
        .map(|v| {
            v.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<Result<_, _>>()?;
    let (channels, height, width) = (input_dims[0], input_dims[1], input_dims[2]);
    let input_data_size = channels * height * width;

    // Parse input values
    let input_values: Vec<F> = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing input values"))??
        .split_whitespace()
        .map(|v| F::from(v.parse::<i64>().expect("Invalid input value")))
        .collect();

    if input_values.len() != input_data_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Input values size mismatch",
        ));
    }

    // Parse y2 header
    let y2_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing y2 header"))??;
    let y2_dims: Vec<usize> = y2_header
        .split_whitespace()
        .skip(2) // Skip "relu y2"
        .take(3)
        .map(|v| {
            v.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<Result<_, _>>()?;
    let q: usize = y2_header
        .split_whitespace()
        .last()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing Q value"))? // Converts Option to Result
        .parse()
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid Q value: {}", e),
            )
        })?; // Handles parsing errors
    let y2_data_size = y2_dims[0] * y2_dims[1] * y2_dims[2];

    // Parse y2 values
    let y2_values: Vec<F> = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing y2 values"))??
        .split_whitespace()
        .map(|v| F::from(v.parse::<i64>().expect("Invalid y2 value")))
        .collect();

    if y2_values.len() != y2_data_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "y2 values size mismatch",
        ));
    }

    // Parse remainder header
    let remainder_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing remainder header"))??;
    let remainder_dims: Vec<usize> = remainder_header
        .split_whitespace()
        .skip(1)
        .map(|v| {
            v.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<Result<_, _>>()?;
    let remainder_data_size = remainder_dims[0] * remainder_dims[1] * remainder_dims[2];

    // Parse remainder values
    let remainder_values: Vec<F> = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing remainder values"))??
        .split_whitespace()
        .map(|v| F::from(v.parse::<i64>().expect("Invalid remainder value")))
        .collect();

    if remainder_values.len() != remainder_data_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Remainder values size mismatch",
        ));
    }

    // Parse output header
    let output_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing output header"))??;
    let output_dims: Vec<usize> = output_header
        .split_whitespace()
        .skip(2)
        .map(|v| {
            v.parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .collect::<Result<_, _>>()?;
    let output_data_size = output_dims[0] * output_dims[1] * output_dims[2];

    // Parse output values
    let output_values: Vec<F> = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing output values"))??
        .split_whitespace()
        .map(|v| F::from(v.parse::<i64>().expect("Invalid output value")))
        .collect();

    if output_values.len() != output_data_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Output values size mismatch",
        ));
    }

    Ok((
        input_values,
        y2_values,
        output_values,
        remainder_values,
        channels,
        height,
        width,
    ))
}
fn benchmark_relu_old_files(c: &mut Criterion) {
    let dir_path = "./dat/dat";
    let relu_files = fs::read_dir(dir_path)
        .expect("Unable to read directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("relu_layer_")
        })
        .collect::<Vec<_>>();

    for entry in relu_files {
        let file_path = entry.path();
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

        let (y1_values, y2_values, y3_values, remainder_values, channels, height, width) =
            read_relu_data(&file_path).expect(&format!("Failed to read file: {}", file_name));

        let q = 6; // Assume q = 6; can be dynamically set based on the file
        let prover = Prover::new_real_data(
            q,
            y1_values.clone(),
            y2_values.clone(),
            y3_values.clone(),
            remainder_values.clone(),
        );
        let verifier = Verifier::new(q, y1_values, y2_values, y3_values, remainder_values);

        // Step 1 pre-computation for Prover
        let (commit_step1, pk_step1, ck_step1, t_step1) =
            prover.process_step1_logup(&prover.remainder, q as usize);

        // Step 1 benchmark for Prover
        c.bench_function(&format!("Prover prove step1 - {}", file_name), |b| {
            b.iter(|| {
                let mut rng = test_rng();
                prover.prove_step1_sumcheck(&mut rng);
                prover.prove_step1_logup(commit_step1.clone(), pk_step1.clone(), t_step1.clone());
            });
        });

        // Pre-computation for Verifier
        let mut rng = test_rng();
        let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
        let (commit_step1, proof_step1, a_step1, t_step1) =
            prover.prove_step1_logup(commit_step1.clone(), pk_step1.clone(), t_step1.clone());

        // Step 1 benchmark for Verifier
        c.bench_function(&format!("Verifier verify step1 - {}", file_name), |b| {
            b.iter(|| {
                verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
                verifier.verify_step1_logup(
                    &commit_step1,
                    &proof_step1,
                    &a_step1,
                    &t_step1,
                    &ck_step1,
                );
            });
        });

        // Step 2 pre-computation for Prover
        let (a_step2, t_step2) = prover.compute_a_t(&prover.y2, &prover.y3);
        let mut transcript = Transcript::new(b"Logup");
        let (commit_step2, pk_step2, ck_step2) = prover.process_step2_logup(&a_step2);

        // Step 2 benchmark for Prover
        c.bench_function(&format!("Prover prove step2 - {}", file_name), |b| {
            b.iter(|| {
                prover.prove_step2_logup(
                    commit_step2.clone(),
                    pk_step2.clone(),
                    t_step2.clone(),
                    a_step2.clone(),
                    &mut transcript,
                );
            });
        });

        // Step 2 pre-computation for Verifier
        let mut transcript = Transcript::new(b"Logup");
        let (commit_step2, pk_step2, ck_step2) = prover.process_step2_logup(&a_step2);

        let (commit_step2, proof_step2, a_step2_p, t_step2_p) = prover.prove_step2_logup(
            commit_step2.clone(),
            pk_step2.clone(),
            t_step2.clone(),
            a_step2.clone(),
            &mut transcript,
        );
        // let mut transcript = Transcript::new(b"Logup");

        // Step 2 benchmark for Verifier
        c.bench_function(&format!("Verifier verify step2 - {}", file_name), |b| {
            b.iter(|| {
                verifier.verify_step2_logup(
                    &commit_step2,
                    &proof_step2,
                    &a_step2_p,
                    &t_step2_p,
                    &ck_step2,
                    // &mut transcript,
                );
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_relu_old_files
}
criterion_main!(benches);
