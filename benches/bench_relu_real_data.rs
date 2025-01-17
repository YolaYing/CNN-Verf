use criterion::{criterion_group, criterion_main, Criterion};
use merlin::Transcript;
use std::fs;
use std::path::Path;
use std::time::Duration;
use zkconv::relu::{prover::Prover, verifier::Verifier};
use zkconv::{relu, E, F};

use ark_ff::{Field, PrimeField, UniformRand};
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    test_rng,
};

fn read_relu_data<P: AsRef<Path>>(file_path: P) -> io::Result<(Vec<F>, Vec<F>)> {
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

    Ok((input_values, output_values))
}

fn benchmark_relu_real_data(c: &mut Criterion) {
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

    // for entry in relu_files {
    let entry = &relu_files[0];
    let file_path = entry.path();
    let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

    let (y1_values, y3_values) =
        read_relu_data(&file_path).expect(&format!("Failed to read file: {}", file_name));

    // let file_name = "./dat/dat/relu_layer_16.txt";
    // let (y1_values, y3_values) =
    //     read_relu_data(&file_name).expect(&format!("Failed to read file: {}", file_name));

    let q = 6; // Assume q = 6; can be dynamically set based on the file
    let prover = Prover::new(q, y1_values.clone(), y3_values.clone());
    let verifier = Verifier::new(q, y1_values, y3_values.clone());
    // Step 1 pre-computation for Prover
    let mut rng = test_rng();
    let r = F::rand(&mut rng);
    let t = prover.compute_table_set(r);
    let a = prover.compute_a(r);

    // preprocess
    let (commit, pk, ck) = prover.process_logup(&a);

    // Step 1 benchmark for Prover
    c.bench_function(&format!("Prover prove - {}", file_name), |b| {
        b.iter(|| {
            // let mut rng = test_rng();
            // let r = F::rand(&mut rng);
            prover.compute_table_set(r);
            prover.compute_a(r);
            prover.prove_logup(commit.clone(), pk.clone(), a.clone(), t.clone());
        });
    });

    // Prove and verify logup for y1 and y3
    let (commit, proof, a, t) = prover.prove_logup(commit, pk, a, t);

    c.bench_function(&format!("Verifier verify - {}", file_name), |b| {
        b.iter(|| verifier.verify_logup(&commit, &proof, &a, &t, &ck));
    });
}
// }

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_relu_real_data
}
criterion_main!(benches);
