use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_std::rc::Rc;
use ark_std::test_rng;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
};
use std::path::Path;
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

fn verify_y2_matches_y1_slices(
    y1: &DenseMultilinearExtension<F>,
    y2: &DenseMultilinearExtension<F>,
    num_vars_y1: usize,
    num_vars_y2: usize,
) -> bool {
    // Total evaluations for y1 and y2
    let y1_values = y1.to_evaluations();
    let y2_values = y2.to_evaluations();

    // Ensure dimensions match expected sizes
    let size_y1 = 1 << num_vars_y1; // Total evaluations in y1
    let size_y2 = 1 << num_vars_y2; // Total evaluations in y2
    assert_eq!(y1_values.len(), size_y1, "Mismatch in y1 size");
    assert_eq!(y2_values.len(), size_y2, "Mismatch in y2 size");

    // Compute stride (number of evaluations per group in y1)
    let stride_b1b2 = 1 << (num_vars_y1 - num_vars_y2);

    // Verify each y2 value
    for sub_i in 0..size_y2 {
        let mut max_val = F::zero();

        // Find the maximum value in the corresponding slice of y1
        for i in 0..stride_b1b2 {
            let index = i * size_y2 + sub_i; // Index in y1 for the current slice
            max_val = max_val.max(y1_values[index]);
        }

        // Compare with the actual value in y2
        if max_val != y2_values[sub_i] {
            println!(
                "Mismatch at index {}: expected {}, found {}",
                sub_i, max_val, y2_values[sub_i]
            );
            return false;
        }
    }

    // All values match
    true
}

fn verify_y2_matches_y1_slices_per_channel(
    y1_poly: &DenseMultilinearExtension<F>,
    y2_poly: &DenseMultilinearExtension<F>,
    num_vars_y1: usize,
    num_vars_y2: usize,
    num_channels: usize,
    data_size_y1: usize,
    data_size_y2: usize,
) -> bool {
    // Ensure y1 and y2 dimensions match expected sizes
    let y1_values = y1_poly.to_evaluations();
    let y2_values = y2_poly.to_evaluations();

    assert_eq!(
        y1_values.len(),
        num_channels * data_size_y1,
        "Mismatch in y1 size"
    );
    assert_eq!(
        y2_values.len(),
        num_channels * data_size_y2,
        "Mismatch in y2 size"
    );

    // Compute stride for y1 and y2
    let stride_b1b2 = data_size_y1 / data_size_y2; // Number of evaluations in y1 per evaluation in y2

    // Iterate over each channel
    for channel in 0..num_channels {
        println!("Channel {}:", channel); // Print current channel
        println!(
            "y1 values for this channel: {:?}",
            &y1_values[channel * data_size_y1..(channel + 1) * data_size_y1]
        ); // Print all y1 values for the channel

        for sub_i in 0..data_size_y2 {
            let mut max_val = F::zero();

            // Find the maximum value in the corresponding slice of y1 for the current channel
            // for i in 0..stride_b1b2 {
            //     let index = channel * data_size_y1 + i * data_size_y2 + sub_i; // Index in y1
            //     max_val = max_val.max(y1_values[index]);
            // }
            let mut y1_slice = Vec::new();
            for i in 0..stride_b1b2 {
                let index = channel * data_size_y1 + i * data_size_y2 + sub_i; // Index in y1
                y1_slice.push(y1_values[index]);
                max_val = max_val.max(y1_values[index]); // Update max_val
            }

            // Print the slice and the computed maximum
            println!(
                "  y1 slice for sub_i {}: {:?}, max: {:?}",
                sub_i, y1_slice, max_val
            );

            // Verify that the maximum value matches the corresponding y2 value
            let y2_index = channel * data_size_y2 + sub_i; // Index in y2
            println!("  y2 value at sub_i {}: {:?}", sub_i, y2_values[y2_index]);
            if max_val != y2_values[y2_index] {
                println!(
                    "Mismatch in channel {} at sub_i {}: expected {:?}, found {:?}",
                    channel, sub_i, max_val, y2_values[y2_index]
                );
                return false;
            }
        }
    }

    // All channels pass verification
    true
}

#[test]
fn test_maxpool_with_real_data() {
    let mut rng = test_rng();
    let file_path = "./dat/dat/maxpool_layer_5.txt";

    let (
        maxpool_in_values,
        maxpool_out_values,
        maxpool_in_channel,
        maxpool_in_data,
        maxpool_out_channel,
        maxpool_out_data,
    ) = read_data_from_file(file_path).expect("Failed to read data file");
    println!(
        "Maxpool input: {} channels, {} data points",
        maxpool_in_channel, maxpool_in_data
    );
    println!(
        "Maxpool output: {} channels, {} data points",
        maxpool_out_channel, maxpool_out_data
    );

    // Calculate necessary parameters
    let num_vars_in_data = maxpool_in_data.next_power_of_two().trailing_zeros() as usize;
    let num_vars_in_channel = maxpool_in_channel.next_power_of_two().trailing_zeros() as usize;
    let num_vars_out_data = maxpool_out_data.next_power_of_two().trailing_zeros() as usize;
    let num_vars_out_channel = maxpool_out_channel.next_power_of_two().trailing_zeros() as usize;
    println!("Number of variables in input data: {}", num_vars_in_data);
    println!(
        "Number of variables in input channel: {}",
        num_vars_in_channel
    );
    println!("Number of variables in output data: {}", num_vars_out_data);
    println!(
        "Number of variables in output channel: {}",
        num_vars_out_channel
    );
    let num_vars_y1 = num_vars_in_data + num_vars_in_channel;
    let num_vars_y2 = num_vars_out_data + num_vars_out_channel;
    println!("Number of variables in y1: {}", num_vars_y1);
    println!("Number of variables in y2: {}", num_vars_y2);

    let y1_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_y1, maxpool_in_values);
    let y2_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_y2, maxpool_out_values);

    let num_channels = maxpool_in_channel; // e.g., 512 for y1
    let data_size_y1 = maxpool_in_data; // e.g., 2*2=4 for y1
    let data_size_y2 = maxpool_out_data; // e.g., 1*1=1 for y2

    let is_valid = verify_y2_matches_y1_slices_per_channel(
        &y1_poly,
        &y2_poly,
        num_vars_y1,
        num_vars_y2,
        num_channels,
        data_size_y1,
        data_size_y2,
    );

    assert!(
        is_valid,
        "Mismatch between y1 slices and y2 values across channels"
    );

    // Reorder dimensions
    let new_order = vec![1, 0];
    let y1_reordered = reorder_variable_groups(
        &y1_poly,
        &[num_vars_in_channel, num_vars_in_data],
        &new_order,
    );
    let y2_reordered = reorder_variable_groups(
        &y2_poly,
        &[num_vars_out_channel, num_vars_out_data],
        &new_order,
    );

    let y1 = Rc::new(y1_reordered);
    let y2 = Rc::new(y2_reordered);

    // Verify that y2 matches the maximum of y1 slices
    assert!(
        verify_y2_matches_y1_slices(&y1, &y2, num_vars_y1, num_vars_y2),
        "Mismatch between y1 slices and y2 values"
    );

    // Prover setup
    let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);

    // Prove using sumcheck
    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut rng);

    let (expanded_y2, combined_y1, a, range, commit, pk, ck) = prover.process_inequalities();

    // Prove inequalities using logup
    let (commit, logup_proof, y2_evaluations, max_y1_evaluations) =
        prover.prove_inequalities(&a, &range, &pk, commit.clone());

    // Verifier setup
    let verifier = Verifier::new(num_vars_y2);

    // Verify sumcheck proof
    assert!(
        verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info),
        "Sumcheck verification failed"
    );

    // Verify logup proof
    assert!(
        verifier.verify_inequalities(
            &commit,
            &logup_proof,
            &y2_evaluations,
            &max_y1_evaluations,
            &ck
        ),
        "Logup verification failed"
    );
}
