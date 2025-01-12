use ark_ff::{BigInt, PrimeField, Zero};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    rc::Rc,
    test_rng,
};
use ark_sumcheck::ml_sumcheck::protocol::prover;
use merlin::Transcript;
use std::path::Path;
use zkconv::{relu::prover::Prover, relu::verifier::Verifier, F};

// MAX_VALUE_IN_Y
const MAX_VALUE_IN_Y: u64 = 65536;

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

// /// Verify whether y3 = ReLU(y2)
// /// ReLU(x) = max(0, x)
// use ark_ff::Field;

// pub fn verify_relu<T: Field>(y2: &[T], y3: &[T]) -> bool {
//     if y2.len() != y3.len() {
//         return false;
//     }

//     for i in 0..y2.len() {
//         if y3[i] != y2[i].max(T::zero()) {
//             println!("Mismatch: y2 = {:?}, y3 = {:?}, i = {:?}", y2[i], y3[i], i);
//             return false;
//         }
//     }
//     true
// }

// use std::collections::HashMap;
// use std::collections::HashSet;
// fn check_data_of_a_and_t(a: &Vec<F>, t: &Vec<F>) {
//     println!("Length of a: {}", a.len());
//     println!("Length of t: {}", t.len());

//     // Check range of a and t
//     println!(
//         "Range of a: min = {:?}, max = {:?}",
//         a.iter().min(),
//         a.iter().max()
//     );
//     println!(
//         "Range of t: min = {:?}, max = {:?}",
//         t.iter().min(),
//         t.iter().max()
//     );

//     // Check unique elements in a and t
//     let unique_a: HashSet<_> = a.iter().collect();
//     let unique_t: HashSet<_> = t.iter().collect();
//     println!("Unique elements in a: {}", unique_a.len());
//     println!("Unique elements in t: {}", unique_t.len());

//     // Check histogram of a and t
//     // let hist_a = calculate_histogram(&a);
//     // let hist_t = calculate_histogram(&t);
//     // println!("Histogram of a: {:?}", hist_a);
//     // println!("Histogram of t: {:?}", hist_t);

//     // Check if a is a subset of t
//     let is_subset = a.iter().all(|x| t.contains(x));
//     println!("Is a a subset of t? {}", is_subset);
//     let missing_from_t: Vec<_> = a.iter().filter(|x| !t.contains(x)).collect();
//     println!("Values in a but not in t: {:?}", missing_from_t);
// }

#[test]
fn test_relu_real_data() {
    let file_path = "./dat/dat/relu_layer_26.txt";
    // 26.28.30

    let (y1_values, y2_values, y3_values, remainder_values, channels, height, width) =
        read_relu_data(file_path).expect("Failed to read data file");

    let q = 6; // Assuming Q value from the file

    // let prover = Prover::new(q, y1_values.clone());
    let prover = Prover::new_real_data(
        q,
        y1_values.clone(),
        y2_values.clone(),
        y3_values.clone(),
        remainder_values.clone(),
    );
    let verifier = Verifier::new(
        q,
        y1_values,
        y2_values.clone(),
        y3_values.clone(),
        remainder_values,
    );

    // Prove and verify using sumcheck
    let mut rng = test_rng();
    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
    assert!(verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info));

    // Prove and verify logup for remainder
    let (commit_step1, pk_step1, ck_step1, t_step1) =
        prover.process_step1_logup(&prover.remainder, q as usize);
    let (commit_step1, proof_step1, a_step1, t_step1) =
        prover.prove_step1_logup(commit_step1, pk_step1, t_step1);
    assert!(verifier.verify_step1_logup(
        &commit_step1,
        &proof_step1,
        &a_step1,
        &t_step1,
        &ck_step1
    ));

    // Prove and verify logup for relu
    let (a, t) = prover.compute_a_t(&prover.y2, &prover.y3);

    let mut transcript = Transcript::new(b"Logup");
    let (commit_step2, pk_step2, ck_step2) = prover.process_step2_logup(&a);

    let (commit_step2, proof_step2, a_step2, t_step2) =
        prover.prove_step2_logup(commit_step2, pk_step2, t, a, &mut transcript);
    let mut transcript = Transcript::new(b"Logup");
    assert!(verifier.verify_step2_logup(
        &commit_step2,
        &proof_step2,
        &a_step2,
        &t_step2,
        &ck_step2,
        &mut transcript
    ));

    println!("ReLU layer verification with real data passed successfully.");
}
