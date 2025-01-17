use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_std::rc::Rc;
use ark_std::test_rng;
use zkconv::conv::prover::Prover;
use zkconv::conv::verifier::{Verifier, VerifierMessage};
use zkconv::F;

fn read_and_prepare_data() -> (Vec<F>, Vec<F>, Vec<F>) {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Read the data from the text files
    let file = File::open("output_data.txt").expect("Unable to open data file");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|line| line.unwrap());

    // Skip the prefix lines (e.g., "X:", "W:", "Y:")
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
    let x_line = skip_prefix(&mut lines, "X:");
    let w_line = skip_prefix(&mut lines, "W:");
    let y_line = skip_prefix(&mut lines, "Y:");

    let x: Vec<F> = parse_numbers(&x_line);
    let w: Vec<F> = parse_numbers(&w_line);
    let y: Vec<F> = parse_numbers(&y_line);

    // Check dimensions
    let x_expected = 4 * 2048; // Assuming 3 channels and padded size 34x34
    let w_expected = 64 * 4 * 128; // Assuming derived dimensions for W
    let y_expected = 64 * 2048; // Assuming derived dimensions for Y

    assert!(
        x.len() == x_expected,
        "Dimension mismatch for X: expected {}, found {}",
        x_expected,
        x.len()
    );
    assert!(
        w.len() == w_expected,
        "Dimension mismatch for W: expected {}, found {}",
        w_expected,
        w.len()
    );
    assert!(
        y.len() == y_expected,
        "Dimension mismatch for Y: expected {}, found {}",
        y_expected,
        y.len()
    );

    (x, w, y)
}

#[test]
fn test_prover_verifier() {
    // Step 1: prapare data
    // for zk input X: 4(3->2^2)*2048(34*34->2^11), using 2 variables representing input channel, 11 variables indexing image
    // for zk input W: 64*4(3->2^2)*128(3*34->2^7), using 6 variables representing output channel, 2 variables representing input channel, 7 variables indexing kernel
    // for zk output Y: 64*2048(34*37 -> 2^11), using 6 variables representing output channel, 11 variables indexing output data
    let (x, w, y) = read_and_prepare_data();

    // Step 2: calculate the number of variables
    let num_vars_j = 6; // output channel j:64 = 2^6
    let num_vars_s = 11; //output position index s:2048 = 2^11
    let num_vars_i = 2; // input channel i:4 = 2^2
    let num_vars_a = 11; // input position index a:2048 = 2^11
    let num_vars_b = 7; // kernal position b:128 = 2^7

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

    // Step 3: mock verifier message
    let mut rng = test_rng();
    let r1_values: Vec<F> = (0..num_vars_j).map(|_| F::rand(&mut rng)).collect();
    let r = F::rand(&mut rng);
    let verifier_msg = VerifierMessage { r1_values, r };

    // Step 4: Prover generate proof
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
    ) = prover.prove(&mut rng, verifier_msg);

    // Step 5: Verifier verify proof

    let result = verifier.verify(
        &proof_s,
        &proof_f,
        &proof_g,
        asserted_s,
        asserted_f,
        asserted_g,
        &poly_info_s,
        &poly_info_f,
        &poly_info_g,
    );

    assert!(result, "Verification failed");
}
