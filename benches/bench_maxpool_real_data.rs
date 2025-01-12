use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
};
use ark_std::{test_rng, UniformRand};
use criterion::{criterion_group, criterion_main, Criterion};
use std::fs;
use std::path::Path;
use std::time::Duration;
use zkconv::{
    maxpool::prover::reorder_variable_groups,
    maxpool::{prover::Prover, verifier::Verifier},
    F,
};

fn read_data_from_file<P: AsRef<Path>>(
    file_path: P,
) -> io::Result<(Vec<F>, Vec<F>, usize, usize, usize, usize)> {
    let file = File::open(file_path)?; // Open the specified file
    let reader = BufReader::new(file); // Create a buffered reader for efficient line-by-line reading

    let mut lines = reader.lines();

    // Parse dimensions of maxpool input
    let maxpool_in_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing input header"))??;
    let maxpool_in_dim: Vec<usize> = maxpool_in_header
        .split_whitespace() // Remove extra spaces and split into tokens
        .skip(3) // Skip the first two tokens (e.g., "max pool in:")
        .map(|v| {
            v.parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid dimension value: {:?}", e),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let maxpool_in_channel = maxpool_in_dim[0]; // Number of input channels
    let maxpool_in_data = maxpool_in_dim[1] * maxpool_in_dim[2]; // Total input data size (height * width)

    // Read maxpool input data
    // let maxpool_in_values: Vec<F> = lines
    //     .by_ref() // Borrow the iterator to continue reading
    //     .take(maxpool_in_channel * maxpool_in_data) // Read the exact number of input values
    //     .flat_map(|line| {
    //         let line = line.unwrap(); // Extract the line as a String
    //         line.split_whitespace() // Split the line into individual numbers
    //             .map(|v| F::from(v.parse::<u32>().expect("Invalid data value"))) // Parse numbers into field elements
    //             .collect::<Vec<F>>() // Collect the parsed numbers into a vector
    //     })
    //     .collect();
    let maxpool_in_values: Vec<F> = {
        // Read the first line containing all the input data
        let line = lines.next().unwrap().unwrap(); // Get the single line and unwrap Result
        line.split_whitespace() // Split the line into individual tokens
            .map(|v| {
                F::from(v.parse::<u32>().expect("Invalid data value")) // Parse each token
            })
            .collect::<Vec<F>>() // Collect the parsed tokens into a vector
    };

    // Parse dimensions of maxpool output
    let maxpool_out_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing output header"))??;
    let maxpool_out_dim: Vec<usize> = maxpool_out_header
        .split_whitespace() // Remove extra spaces and split into tokens
        .skip(3) // Skip the first three tokens (e.g., "max pooling output:")
        .map(|v| {
            v.parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid dimension value: {:?}", e),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let maxpool_out_channel = maxpool_out_dim[0]; // Number of output channels
    let maxpool_out_data = maxpool_out_dim[1] * maxpool_out_dim[2]; // Total output data size (height * width)

    // Read maxpool output data
    let maxpool_out_values: Vec<F> = {
        // Read the second line containing all the output data
        let line = lines.next().unwrap().unwrap(); // Get the single line and unwrap Result
        line.split_whitespace() // Split the line into individual tokens
            .map(|v| {
                F::from(v.parse::<u32>().expect("Invalid data value")) // Parse each token
            })
            .collect::<Vec<F>>() // Collect the parsed tokens into a vector
    };

    // Return parsed data and dimensions
    Ok((
        maxpool_in_values,
        maxpool_out_values,
        maxpool_in_channel,
        maxpool_in_data,
        maxpool_out_channel,
        maxpool_out_data,
    ))
}
fn benchmark_maxpool_files(c: &mut Criterion) {
    let dir_path = "./dat/dat";
    let maxpool_files = fs::read_dir(dir_path)
        .expect("Unable to read directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("maxpool_layer_")
        })
        .collect::<Vec<_>>();

    for entry in maxpool_files {
        let file_path = entry.path();
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

        let (
            maxpool_in_values,
            maxpool_out_values,
            maxpool_in_channel,
            maxpool_in_data,
            maxpool_out_channel,
            maxpool_out_data,
        ) = read_data_from_file(&file_path).expect(&format!("Failed to read file: {}", file_name));

        let num_vars_in_data = maxpool_in_data.next_power_of_two().trailing_zeros() as usize;
        let num_vars_in_channel = maxpool_in_channel.next_power_of_two().trailing_zeros() as usize;
        let num_vars_out_data = maxpool_out_data.next_power_of_two().trailing_zeros() as usize;
        let num_vars_out_channel =
            maxpool_out_channel.next_power_of_two().trailing_zeros() as usize;

        let num_vars_y1 = num_vars_in_data + num_vars_in_channel;
        let num_vars_y2 = num_vars_out_data + num_vars_out_channel;

        let y1_poly =
            DenseMultilinearExtension::from_evaluations_vec(num_vars_y1, maxpool_in_values.clone());
        let y2_poly = DenseMultilinearExtension::from_evaluations_vec(
            num_vars_y2,
            maxpool_out_values.clone(),
        );

        let new_order = vec![4, 1, 3, 0, 2];
        let y1_reordered = reorder_variable_groups(
            &y1_poly,
            &[
                1,
                (num_vars_in_data - 2) / 2,
                1,
                (num_vars_in_data - 2) / 2,
                num_vars_in_channel,
            ],
            &new_order,
        );

        let new_order = vec![1, 0];
        let y2_reordered = reorder_variable_groups(
            &y2_poly,
            &[num_vars_out_data, num_vars_out_channel],
            &new_order,
        );

        let y1 = Rc::new(y1_reordered);
        let y2 = Rc::new(y2_reordered);

        // Prover setup
        let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);
        let (expanded_y2, combined_y1, a, range, commit, pk, ck) = prover.process_inequalities();

        c.bench_function(
            &format!("Maxpool Prover total prove - {}", file_name),
            |b| {
                b.iter(|| {
                    let mut rng = test_rng();
                    prover.prove_sumcheck(&mut rng);
                    prover.prove_inequalities(&a, &range, &pk, commit.clone());
                });
            },
        );

        let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut test_rng());
        let (commit_logup, logup_proof, a_proof, range_proof) =
            prover.prove_inequalities(&a, &range, &pk, commit.clone());

        // Verifier setup
        let verifier = Verifier::new(num_vars_y2);

        c.bench_function(
            &format!("Maxpool Verifier total verification - {}", file_name),
            |b| {
                b.iter(|| {
                    verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
                    verifier.verify_inequalities(
                        &commit_logup,
                        &logup_proof,
                        &a_proof,
                        &range_proof,
                        &ck,
                    );
                });
            },
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = benchmark_maxpool_files
}
criterion_main!(benches);
