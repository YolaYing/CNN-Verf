use ark_ff::{PrimeField, UniformRand};
use ark_std::rand::thread_rng;
use merlin::Transcript;
use poly_iop::perm_check::PermCheck;
use std::collections::VecDeque;
use zkconv::conv_padding_old::prover::Prover;
use zkconv::conv_padding_old::verifier::Verifier;
use zkconv::{E, F};

fn read_and_prepare_data() -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Read the data from the text files
    let file = File::open("pad_and_rot_data.txt").expect("Unable to open data file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|line| line.unwrap());

    // Skip the prefix lines (e.g., "X:", "Y:")
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

    // Helper function to parse a line of space-separated numbers into a vector of F
    fn parse_numbers(line: &str) -> Vec<F> {
        line.split_whitespace() // Split the line into whitespace-separated tokens
            .filter_map(|token| {
                // Attempt to parse each token, logging any failures
                match token.parse::<F>() {
                    Ok(value) => Some(value),
                    Err(_) => {
                        eprintln!("Warning: Invalid number detected: {}", token);
                        None
                    }
                }
            })
            .collect()
    }

    // Extract data for X, W, and Y
    let x_line = skip_prefix(&mut lines, "x:");
    let x_padded_line = skip_prefix(&mut lines, "x_padded:");
    let y_line = skip_prefix(&mut lines, "y:");
    let y_real_line = skip_prefix(&mut lines, "y_real:");

    let x: Vec<F> = parse_numbers(&x_line);
    let x_padded: Vec<F> = parse_numbers(&x_padded_line);
    let y: Vec<F> = parse_numbers(&y_line);
    let y_real: Vec<F> = parse_numbers(&y_real_line);

    // Check dimensions
    let x_expected = 3 * 32 * 32; // Assuming 3 channels and original size 32x32
    let x_padded_expected = 3 * 34 * 34; // Assuming 3 channels and padded size 34x34
    let y_expected = 64 * 34 * 37; // Assuming 64 channels and original size 32x32
    let y_real_expected = 64 * 32 * 32; // Assuming 64 channels and original size 32x32

    assert!(
        x.len() == x_expected,
        "Dimension mismatch for X: expected {}, found {}",
        x_expected,
        x.len()
    );
    assert!(
        x_padded.len() == x_padded_expected,
        "Dimension mismatch for X_padded: expected {}, found {}",
        x_padded_expected,
        x_padded.len()
    );
    assert!(
        y.len() == y_expected,
        "Dimension mismatch for Y: expected {}, found {}",
        y_expected,
        y.len()
    );
    assert!(
        y_real.len() == y_real_expected,
        "Dimension mismatch for Y_real: expected {}, found {}",
        y_real_expected,
        y_real.len()
    );

    (x, x_padded, y, y_real)
}

#[test]
fn test_padding_and_rotation_with_perm_check() {
    let mut rng = thread_rng();

    let (x, x_padded, y, y_real) = read_and_prepare_data();

    let padding = 1; // Assuming a padding of 1
    let kernel_size = 3; // Assuming a kernel size of 3
    let input_channels = 3; // Assuming 3 input channels
    let output_channels = 64; // Assuming 64 output channels

    let prover = Prover::new(
        x,
        x_padded,
        y,
        y_real,
        padding,
        kernel_size,
        input_channels,
        output_channels,
    );
    let verifier = Verifier::new(34 * 34); // Number of variables based on padded size

    let verifier_randomness = F::rand(&mut rng);

    // Prove padding correctness
    let (proof_padding, h_ori_padding, h_padded) =
        prover.prove_padding(&mut rng, verifier_randomness);

    // Verify padding correctness
    let padding_verified = verifier.verify(
        h_ori_padding.clone(),
        h_padded.clone(),
        VecDeque::from(proof_padding),
    );

    assert!(
        padding_verified,
        "Permutation check for padding process failed"
    );

    // Prove rotation correctness
    let (proof_rotation, h_real_rotation, h_calculated) =
        prover.prove_rotation(&mut rng, verifier_randomness);

    // Verify rotation correctness
    let rotation_verified = verifier.verify(
        h_real_rotation.clone(),
        h_calculated.clone(),
        VecDeque::from(proof_rotation),
    );

    assert!(
        rotation_verified,
        "Permutation check for rotation process failed"
    );
}
