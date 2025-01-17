use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    test_rng,
};
use std::path::Path;
use zkconv::{
    relu::{prover::Prover, verifier::Verifier},
    E, F,
};

use ark_ff::{Field, PrimeField, UniformRand};

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

// test if all y3 = relu(y1/ 2^Q)
pub fn test_relu_relationship(Q: u64, y1: Vec<F>, y3: Vec<F>) -> bool {
    let shift_factor = F::from(2u64).pow(&[Q]);

    for i in 0..y1.len() {
        let y1_val = y1[i];
        let y3_val = y3[i];

        // attention: F::from(3246u64) / F::from(64u64)) != F::from(3246u64 / 64u64))
        let shifted_val = F::from(
            y1_val.into_bigint().as_ref()[0] as i64 / shift_factor.into_bigint().as_ref()[0] as i64,
        );

        // attension: after a negative number become field element, it is no loger negative
        // so we need to check if the shifted_val is greater than 65536, if so it used to be a negative number
        let relu_shifted_val = if shifted_val <= F::from(65536u64) {
            shifted_val
        } else {
            F::from(0u64)
        };

        if y3_val != relu_shifted_val {
            println!(
                "Mismatch at index {}: y1 = {}, y3 = {}, relu(y1/2^Q) = {}, shifted_val = {}",
                i, y1_val, y3_val, relu_shifted_val, shifted_val
            );
            println!("3246/64 = {}", F::from(3246u64) / F::from(64u64));
            println!("3246/64 = {}", F::from(3246u64 / 64u64));
            return false;
        }
    }
    true
}

#[test]
fn test_relu_real_data() {
    let file_path = "./dat/dat/relu_layer_26.txt";
    // 23

    let (y1_values, y3_values) = read_relu_data(file_path).expect("Failed to read data file");
    // if test_relu_relationship(6, y1_values.clone(), y3_values.clone()) {
    //     println!("ReLU layer verification with real data passed successfully.");
    // } else {
    //     println!("ReLU layer verification with real data failed.");
    // }

    let q = 6; // Assuming Q value from the file

    // let prover = Prover::new(q, y1_values.clone());
    let prover = Prover::new(q, y1_values.clone(), y3_values.clone());
    let verifier = Verifier::new(q, y1_values, y3_values.clone());

    let mut rng = test_rng();
    let r = F::rand(&mut rng);
    let t = prover.compute_table_set(r);
    let a = prover.compute_a(r);

    // // test if all a in t, if all a in t, print "all a in t", otherwise print "not all a in t"
    // let mut all_a_in_t = true;
    // for a_i in a.iter() {
    //     if !t.contains(a_i) {
    //         all_a_in_t = false;
    //         println!("a not in t: {}", a_i);
    //     }
    // }
    // if all_a_in_t {
    //     println!("all a in t");
    // } else {
    //     println!("not all a in t");
    // }
    // println!("a.max = {}", a.iter().max().unwrap());

    // preprocess
    let (commit, pk, ck) = prover.process_logup(&a);

    // Prove and verify logup for y1 and y3
    let (commit, proof, a, t) = prover.prove_logup(commit, pk, a, t);
    assert!(verifier.verify_logup(&commit, &proof, &a, &t, &ck));

    println!("ReLU layer verification with real data passed successfully.");
}
