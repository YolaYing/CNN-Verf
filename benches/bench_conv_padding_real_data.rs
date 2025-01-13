use ark_ff::{Field, UniformRand};
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    rc::Rc,
    test_rng,
};
use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::Duration;
use zkconv::conv_padding::prover::Prover;
use zkconv::conv_padding::verifier::Verifier;
use zkconv::{E, F};

// Helper functions to parse data
fn parse_dimensions(dim_str: &str) -> Vec<usize> {
    dim_str
        .trim_matches(&['(', ')'][..])
        .split(',')
        .map(|v| v.trim().parse::<usize>().expect("Invalid dimension"))
        .collect()
}

fn parse_line(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }
    let parts: Vec<&str> = line.splitn(2, '[').collect();
    if parts.len() != 2 {
        panic!(
            "Invalid format: expected dimensions and data for '{}'",
            prefix
        );
    }

    let dim_part = parts[0]
        .split(':')
        .nth(1)
        .expect("Missing dimensions")
        .trim();
    let dims = parse_dimensions(dim_part);

    let data_part = parts[1].trim_end_matches(']').trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (dims, values)
}

fn parse_line_with_shapes(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }

    let parts: Vec<&str> = line.split("padded shape:").collect();
    if parts.len() != 2 {
        panic!("Invalid format: expected padded shape in '{}'", line);
    }

    let useful_shape = parts[0]
        .split("useful shape:")
        .nth(1)
        .expect("Missing useful shape")
        .trim();
    let padded_shape = parts[1]
        .split('[')
        .nth(0)
        .expect("Missing padded shape")
        .trim();

    let useful_dims = parse_dimensions(useful_shape);
    let padded_dims = parse_dimensions(padded_shape);

    if useful_dims.len() != padded_dims.len() {
        panic!("Mismatch between useful and padded dimensions");
    }

    let data_part = parts[1]
        .split('[')
        .nth(1)
        .expect("Missing data section")
        .trim_end_matches(']')
        .trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (padded_dims, values)
}

fn read_and_prepare_data<P: AsRef<std::path::Path>>(
    file_path: P,
) -> io::Result<(
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    usize,
    usize,
    usize,
    usize,
)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    let plain_x_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing plain x line"))??;
    let (plain_x_dims, plain_x_values) = parse_line(&plain_x_line, "plain x:");

    let rot_pad_x_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot pad x line"))??;
    let (rot_pad_x_dims, rot_pad_x_values) = parse_line(&rot_pad_x_line, "rot pad x:");

    let weight_w_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing weight W line"))??;
    let (weight_w_dims, weight_w_values) = parse_line(&weight_w_line, "weight W:");

    let conv_y_line = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Missing conv direct Y line")
    })??;
    let (conv_y_dims, conv_y_values) = parse_line(&conv_y_line, "conv direct Y:");

    let rot_y_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot y line"))??;
    let (rot_y_dims, rot_y_values) = parse_line_with_shapes(&rot_y_line, "rot y:");

    let p_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing P line"))??;
    let (p_dims, p_values) = parse_line_with_shapes(&p_line, "P:");

    let padding = 1;
    let kernel_size = (weight_w_dims[2] as f64).sqrt() as usize;
    let input_channels = plain_x_dims[0];
    let output_channels = conv_y_dims[0];

    Ok((
        plain_x_values,
        rot_pad_x_values,
        conv_y_values,
        rot_y_values,
        p_values,
        padding,
        kernel_size,
        input_channels,
        output_channels,
    ))
}

fn benchmark_conv_files(c: &mut Criterion) {
    let dir_path = "./dat/dat";
    let conv_files = fs::read_dir(dir_path)
        .expect("Unable to read directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("conv_layer_")
        })
        .collect::<Vec<_>>();

    for entry in conv_files {
        let file_path = entry.path();
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

        let (x, x_padded, y, y_real, p, padding, kernel_size, input_channels, output_channels) =
            read_and_prepare_data(&file_path)
                .expect(&format!("Failed to read file: {}", file_name));

        let mut rng = test_rng();

        // Prover setup
        let prover = Prover::new(
            x.clone(),
            x_padded.clone(),
            y.clone(),
            y_real.clone(),
            p.clone(),
            padding,
            kernel_size,
            input_channels,
            output_channels,
        );

        let verifier = Verifier::new(&x_padded.len() / input_channels);

        // Benchmark Prover
        c.bench_function(&format!("Conv Prover - {}", file_name), |b| {
            b.iter(|| {
                let verifier_randomness = F::rand(&mut rng);
                let (proof_padding, h_ori_padding, h_padded) =
                    prover.prove_padding(&mut rng, verifier_randomness);

                let verifier_randomness: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();
                let (proof_rotation, h_real_rotation, h_calculated) =
                    prover.prove_rotation(&mut rng, verifier_randomness);
            });
        });

        // Generate proofs for verification
        let verifier_randomness = F::rand(&mut rng);
        let (proof_padding, h_ori_padding, h_padded) =
            prover.prove_padding(&mut rng, verifier_randomness);

        let verifier_randomness: Vec<F> = (0..2).map(|_| F::rand(&mut rng)).collect();
        let (proof_rotation, h_real_rotation, h_calculated) =
            prover.prove_rotation(&mut rng, verifier_randomness);

        // Benchmark Verifier
        c.bench_function(&format!("Conv Verifier - {}", file_name), |b| {
            b.iter(|| {
                let padding_verified = verifier.verify(
                    h_ori_padding.clone(),
                    h_padded.clone(),
                    VecDeque::from(proof_padding.clone()),
                );

                let rotation_verified = verifier.verify(
                    h_real_rotation.clone(),
                    h_calculated.clone(),
                    VecDeque::from(proof_rotation.clone()),
                );
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_conv_files
}
criterion_main!(benches);
