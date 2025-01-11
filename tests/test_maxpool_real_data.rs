use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_std::rc::Rc;
use ark_std::test_rng;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
};
use rand_chacha::rand_core::le;
use std::num::ParseIntError;
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
            if max_val != y2_values[y2_index] {
                println!(
                    "Mismatch in channel {} at sub_i {}: expected {:?}, found {:?}, y2 index {:?}",
                    channel, sub_i, max_val, y2_values[y2_index], y2_index,
                );
                println!(
                    "index for y1 slices: {:?}, {:?}, {:?}, {:?}, channel: {:?}, data_size_y1: {:?}, data_size_y2: {:?}, sub_i: {:?}",
                    channel * data_size_y1 + 0 * data_size_y2 + sub_i,
                    channel * data_size_y1 + 1 * data_size_y2 + sub_i,
                    channel * data_size_y1 + 2 * data_size_y2 + sub_i,
                    channel * data_size_y1 + 3 * data_size_y2 + sub_i,
                    channel,
                    data_size_y1,
                    data_size_y2,
                    sub_i,
                );
                // print y2 values in this channel
                for i in 0..data_size_y2 {
                    println!(
                        "  y2 value at sub_i {}: {:?}",
                        i,
                        y2_values[channel * data_size_y2 + i]
                    );
                }
                return false;
            }
        }
    }

    // All channels pass verification
    true
}

/// Verify the maxpool constraints based on Python logic
/// Parameters:
/// - `in_values`: DenseMultilinearExtension for input values
/// - `out_values`: DenseMultilinearExtension for output values
/// - `maxpool_in_channel`: Number of input channels
/// - `maxpool_in_data`: Size of input data per channel
/// - `maxpool_out_data`: Size of output data per channel
///
/// Returns:
/// - `Result<(), String>`: Ok if the data passes the constraints, Err with an error message otherwise
fn verify_maxpool_constraints(
    in_values: &DenseMultilinearExtension<F>,
    out_values: &DenseMultilinearExtension<F>,
    maxpool_in_channel: usize,
    maxpool_in_data: usize,
    maxpool_out_data: usize,
) -> Result<(), String> {
    // Convert DenseMultilinearExtension values to vectors
    let in_values = in_values.to_evaluations();
    let out_values = out_values.to_evaluations();

    // Ensure input and output sizes are consistent with constraints
    if in_values.len() != maxpool_in_channel * maxpool_in_data {
        return Err(format!(
            "Invalid input size: expected {}, got {}",
            maxpool_in_channel * maxpool_in_data,
            in_values.len()
        ));
    }
    if out_values.len() != maxpool_in_channel * maxpool_out_data {
        return Err(format!(
            "Invalid output size: expected {}, got {}",
            maxpool_in_channel * maxpool_out_data,
            out_values.len()
        ));
    }

    // Verify constraints
    for co in 0..maxpool_in_channel {
        for x in 0..maxpool_out_data {
            let i = x / 16;
            let j = x % 16;

            // Extract the 4 values to be maxpooled
            let s = [
                in_values[co * maxpool_in_data + 32 * 2 * i + j * 2],
                in_values[co * maxpool_in_data + 32 * 2 * i + j * 2 + 1],
                in_values[co * maxpool_in_data + 32 * (2 * i + 1) + j * 2],
                in_values[co * maxpool_in_data + 32 * (2 * i + 1) + j * 2 + 1],
            ];

            // Compute the maximum value
            let max_value = *s.iter().max().unwrap();
            // print channel, x, index of invalues, value of invalues, max_value, index of outvalues, value of outvalues
            if co == 2 {
                println!(
                    "channel: {}, x: {}, in index1: {}, index2: {}, index3: {}, index4:{}, in values: {:?}, max_value: {}, out index: {}, value: {}",
                    co,
                    x,
                    co * maxpool_in_data + 32 * 2 * i + j * 2,
                    co * maxpool_in_data + 32 * 2 * i + j * 2 + 1,
                    co * maxpool_in_data + 32 * (2 * i + 1) + j * 2,
                    co * maxpool_in_data + 32 * (2 * i + 1) + j * 2 + 1,
                    s,
                    max_value,
                    co * maxpool_out_data + x,
                    out_values[co * maxpool_out_data + x]
                );
            }

            // Check against the output value
            if max_value != out_values[co * maxpool_out_data + x] {
                return Err(format!(
                    "Mismatch at channel {} and index {}: expected {}, found {}",
                    co,
                    x,
                    max_value,
                    out_values[co * maxpool_out_data + x]
                ));
            }
        }
    }

    // All checks passed
    Ok(())
}

#[test]
fn test_verify_maxpool_constraints() {
    let file_path = "./dat/dat/maxpool_layer_5.txt";

    let (
        maxpool_in_values,
        maxpool_out_values,
        maxpool_in_channel,
        maxpool_in_data,
        maxpool_out_channel,
        maxpool_out_data,
    ) = read_data_from_file(file_path).expect("Failed to read data file");

    //  let y1_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_y1, maxpool_in_values);
    // let y2_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_y2, maxpool_out_values);

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

    let in_values = DenseMultilinearExtension::from_evaluations_vec(num_vars_y1, maxpool_in_values);
    let out_values =
        DenseMultilinearExtension::from_evaluations_vec(num_vars_y2, maxpool_out_values);

    match verify_maxpool_constraints(
        &in_values,
        &out_values,
        maxpool_in_channel,
        maxpool_in_data,
        maxpool_out_data,
    ) {
        Ok(_) => println!("All constraints verified successfully!"),
        Err(err) => panic!("Verification failed: {}", err),
    }
}

fn verify_y2_matches_y1(
    y1: &DenseMultilinearExtension<F>,
    y2: &DenseMultilinearExtension<F>,
    num_b2b3: usize, // number of (b2, b3, ...) groups
    num_channels: usize,
) -> bool {
    for channel in 0..num_channels {
        for b2b3 in 0..(1 << num_b2b3) {
            let mut max_val = F::zero();
            for i in 0..2 {
                for j in 0..2 {
                    let idx_y1 = i * (1 << (num_b2b3 + num_channels + 2 - 1))
                        + j * (1 << (num_b2b3 + num_channels + 2 - 2))
                        + b2b3 * (1 << num_channels)
                        + channel;
                    max_val = max_val.max(y1[idx_y1].clone());

                    if channel == 2 {
                        println!(
                            "channel: {}, b2b3: {}, i: {}, j: {}, idx_y1: {}, value: {}",
                            channel, b2b3, i, j, idx_y1, y1[idx_y1]
                        );
                    }
                }
            }

            let idx_y2 = b2b3 * (1 << num_channels) + channel; // 只用 b2, b3, channel 拼接
            if channel == 2 {
                println!(
                    "channel: {}, b2b3: {}, idx_y2: {}, value: {}",
                    channel, b2b3, idx_y2, y2[idx_y2]
                );
            }
            if max_val != y2[idx_y2] {
                println!(
                    "Mismatch for (channel: {}, b2b3: {}): expected {:?}, found {:?}",
                    channel, b2b3, max_val, y2[idx_y2]
                );
                return false;
            }
        }
    }
    true
}

fn switch_to_little_endian_and_convert(
    num_vars: usize,
    values: &Vec<F>,
) -> DenseMultilinearExtension<F> {
    // Total number of elements based on the number of variables
    let num_elements = 1 << num_vars;

    // Check the input size
    assert_eq!(
        values.len(),
        num_elements,
        "Values size does not match the number of variables."
    );

    // Convert indices to little-endian
    let mut reordered_values = vec![F::zero(); num_elements];
    for (index, &value) in values.into_iter().enumerate() {
        let little_endian_index = reverse_bits(index as u32, num_vars) as usize;
        reordered_values[little_endian_index] = value;
    }

    // Create DenseMultilinearExtension with reordered values
    DenseMultilinearExtension::from_evaluations_vec(num_vars, reordered_values)
}

/// Reverse the bits of an index up to the given number of variables
fn reverse_bits(index: u32, num_vars: usize) -> u32 {
    let mut reversed = 0;
    for i in 0..num_vars {
        if (index & (1 << i)) != 0 {
            reversed |= 1 << (num_vars - 1 - i);
        }
    }
    reversed
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

    // check maxpool_in_values.len() = 2^num_vars_y1
    // check maxpool_out_values.len() = 2^num_vars_y2
    assert!(
        maxpool_in_values.len() == (1 << num_vars_y1),
        "Dimension mismatch for maxpool_in_values: expected {}, found {}",
        1 << num_vars_y1,
        maxpool_in_values.len()
    );
    assert!(
        maxpool_out_values.len() == (1 << num_vars_y2),
        "Dimension mismatch for maxpool_out_values: expected {}, found {}",
        1 << num_vars_y2,
        maxpool_out_values.len()
    );

    // Switch the input and output values to little-endian form

    let y1_poly =
        DenseMultilinearExtension::from_evaluations_vec(num_vars_y1, maxpool_in_values.clone());
    let y2_poly =
        DenseMultilinearExtension::from_evaluations_vec(num_vars_y2, maxpool_out_values.clone());

    // Convert indices and generate polynomials
    // let y1_poly = switch_to_little_endian_and_convert(num_vars_y1, &maxpool_in_values);
    // let y2_poly = switch_to_little_endian_and_convert(num_vars_y2, &maxpool_out_values);
    // compare all y1_poly with maxpool_in_values

    // Reorder dimensions
    // let new_order = vec![1, 0];
    // let y1_reordered = reorder_variable_groups(
    //     &y1_poly,
    //     &[num_vars_in_channel, num_vars_in_data],
    //     &new_order,
    // );
    // let y2_reordered = reorder_variable_groups(
    //     &y2_poly,
    //     &[num_vars_out_channel, num_vars_out_data],
    //     &new_order,
    // );
    // let new_order = vec![4, 2, 0, 3, 1];
    // let y1_reordered = reorder_variable_groups(
    //     &y1_poly,
    //     &[
    //         num_vars_in_channel,
    //         (num_vars_in_data - 2) / 2,
    //         1,
    //         (num_vars_in_data - 2) / 2,
    //         1,
    //     ],
    //     &new_order,
    // );

    // let new_order = vec![1, 0];
    // let y2_reordered = reorder_variable_groups(
    //     &y2_poly,
    //     &[num_vars_out_channel, num_vars_out_data],
    //     &new_order,
    // );

    // define new order：new group i = old group new_order[i]
    // let new_order = vec![0, 3, 1, 4, 2];
    // let new_order = vec![0, 2, 4, 1, 3];
    // switch i, j to the lowest bits
    // let new_order = vec![2, 4, 1, 3, 0];
    // let new_order = vec![0, 3, 1, 4, 2];
    // num_vars_in_channel, (num_vars_in_data - 2) / 2, 1, (num_vars_in_data - 2) / 2, 1
    // -> 1,1,(num_vars_in_data - 2) / 2,(num_vars_in_data - 2) / 2,num_vars_in_channel
    let new_order = vec![4, 1, 3, 0, 2];
    let y1_reordered = reorder_variable_groups(
        &y1_poly,
        // &[
        //     num_vars_in_channel,
        //     (num_vars_in_data - 2) / 2,
        //     1,
        //     (num_vars_in_data - 2) / 2,
        //     1,
        // ],
        &[
            1,
            (num_vars_in_data - 2) / 2,
            1,
            (num_vars_in_data - 2) / 2,
            num_vars_in_channel,
        ],
        &new_order,
    );
    println!("y1_reordered[4482]: {:?}", y1_reordered[4482]);
    // let index = 0*2usize.pow(1+1+14)

    // switch num_vars_in_data to the lowest bits
    let new_order = vec![1, 0];
    let y2_reordered = reorder_variable_groups(
        &y2_poly,
        &[num_vars_out_data, num_vars_out_channel],
        &new_order,
    );

    let y1 = Rc::new(y1_reordered);
    let y2 = Rc::new(y2_reordered);

    // Verify that y2 matches the maximum of y1 slices
    let num_channels = num_vars_in_channel;
    let num_b2b3 = num_vars_out_data;
    println!("num_channels: {}, num_b2b3: {}", num_channels, num_b2b3);

    let is_valid = verify_y2_matches_y1(&y1, &y2, num_b2b3, num_channels);

    assert!(
        is_valid,
        "Mismatch between y1 slices and y2 values across channels"
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

#[test]
fn test_reverse_bits() {
    // Case 1: Basic example
    let index = 0b0101; // Decimal: 5
    let num_vars = 4;
    let reversed = reverse_bits(index, num_vars);
    assert_eq!(reversed, 0b1010); // Expected: 10」
}
