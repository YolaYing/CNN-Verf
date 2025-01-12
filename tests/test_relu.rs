use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merlin::Transcript;

/// Generate mock data for testing the relu layer proof setup.
/// 1. We randomly generate y1 as a vector of field elements interpreted as integers.
/// 2. We compute y2 = floor(y1 / 2^Q) by shifting right Q bits (assuming y1 is interpreted as an integer).
/// 3. We compute remainder = y1 - 2^Q * y2.
/// 4. We compute y3 = relu(y2) = max(y2,0).
///
/// Note: Since F is a finite field, interpreting it directly as an integer might be nontrivial,
/// but for mock test data we can assume y1 is small enough to fit in a u64, and that we treat
/// them as unsigned integers for simplicity. In a real scenario, additional care is needed for
/// handling negative values or mapping field elements to integers.
pub fn generate_mock_data(Q: u32, length: usize) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
    let mut rng = test_rng();

    // 1. Generate y1 as random integers and map to field. For simplicity, assume small values.
    let y1_ints: Vec<u64> = (0..length)
        .map(|_| rng.next_u64() % (1 << (Q + 10)))
        .collect();
    let y1: Vec<F> = y1_ints.iter().map(|&x| F::from(x)).collect();

    // 2. Compute y2 = y1 >> 2^Q (as integers)
    //    y2_i = floor(y1_i / 2^Q)
    let two_pow_q = 1u64 << Q;
    let y2_ints: Vec<u64> = y1_ints.iter().map(|&x| x >> Q).collect();
    let y2: Vec<F> = y2_ints.iter().map(|&x| F::from(x)).collect();

    // 3. remainder = y1 - 2^Q * y2
    //    remainder_i = y1_i - (y2_i << Q)
    let remainder_ints: Vec<u64> = y1_ints
        .iter()
        .zip(y2_ints.iter())
        .map(|(&y1_i, &y2_i)| y1_i - (y2_i << Q))
        .collect();
    let remainder: Vec<F> = remainder_ints.iter().map(|&x| F::from(x)).collect();

    // 4. y3 = relu(y2)
    //    relu(y2_i) = y2_i if y2_i >=0 else 0
    // Here we assume all generated values are non-negative. If negative values were needed,
    // we would have to define how we represent negative integers in the field.
    // For simplicity, consider all generated values non-negative and so relu(y2)=y2.
    // If needed, you could add logic:
    // let y3 = y2_ints.iter().map(|&x| if (x as i64) < 0 { F::from(0u64) } else { F::from(x) }).collect();
    let y3 = y2.clone();

    (y1, y2, y3, remainder)
}

use zkconv::relu::{prover::Prover, verifier::Verifier};
use zkconv::{E, F}; // Assuming the function is defined in a helper file or above test functions

#[test]
fn test_full_prove_and_verify_with_mock_data() {
    let Q: u32 = 4;
    let length = 16;
    let (y1, y2, y3, remainder) = generate_mock_data(Q, length);

    // Initialize Prover and Verifier with these mock data:
    // The Prover normally computes y2, y3, remainder internally, but here we can also verify
    // that the Prover matches our computed mock data if needed. For simplicity, we just
    // instantiate Prover with y1 and let it do its job. Then we compare results later if desired.

    let prover = Prover::new(Q, y1.clone());
    let verifier = Verifier::new(
        Q,
        y1.clone(),
        prover.y2.clone(),
        prover.y3.clone(),
        prover.remainder.clone(),
    );

    // Proceed with sumcheck and logup proofs as in previous tests
    println!("Testing sumcheck proof");
    let mut rng = ark_std::test_rng();
    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
    assert!(verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info));

    // println!("Testing logup remainder range proof");
    // let (commit_step1, proof_step1, a_step1, t_step1) = prover.prove_step1_logup();
    // assert!(verifier.verify_step1_logup(&commit_step1, &proof_step1, &a_step1, &t_step1));

    // println!("Testing logup relu proof");
    // let (commit_step2, proof_step2, a_step2, t_step2) = prover.prove_step2_logup();
    // assert!(verifier.verify_step2_logup(&commit_step2, &proof_step2, &a_step2, &t_step2));

    let (commit_step1, pk_step1, ck_step1, t_step1) =
        prover.process_step1_logup(&prover.remainder, prover.Q as usize);

    // Step 1: Logup remainder range proof
    let (commit_step1, proof_step1, a_step1, t_step1) =
        prover.prove_step1_logup(commit_step1, pk_step1, t_step1);
    assert!(verifier.verify_step1_logup(
        &commit_step1,
        &proof_step1,
        &a_step1,
        &t_step1,
        &ck_step1
    ));

    // Step 2: Logup relu proof
    // let (commit_step2, pk_step2, ck_step2, t_step2) =
    //     prover.process_logup(&prover.y3, prover.Q as usize);
    // let (commit_step2, proof_step2, a_step2, t_step2) =
    //     prover.prove_step2_logup(commit_step2, pk_step2, t_step2);
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

    println!("Test with mock data passed successfully");
}

// use ark_ff::UniformRand;
// use ark_std::{test_rng, vec::Vec};
// use zkconv::relu::prover::Prover;
// use zkconv::relu::verifier::Verifier;
// use zkconv::{E, F};

// fn generate_test_data(len: usize) -> Vec<F> {
//     let mut rng = test_rng();
//     (0..len).map(|_| F::rand(&mut rng)).collect()
// }

// #[test]
// fn test_full_prove_and_verify() {
//     let mut rng = test_rng();

//     let Q: u32 = 4;
//     let num_elements = 16;

//     let y1 = generate_test_data(num_elements);
//     let prover = Prover::new(Q, y1.clone());
//     let verifier = Verifier::new(
//         Q,
//         y1.clone(),
//         prover.y2.clone(),
//         prover.y3.clone(),
//         prover.remainder.clone(),
//     );

//     // Step 1: sumcheck
//     let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_step1_sumcheck(&mut rng);
//     let result_sumcheck = verifier.verify_step1_sumcheck(&sumcheck_proof, asserted_sum, &poly_info);
//     assert!(result_sumcheck, "Step 1: Sumcheck verification failed");

//     // Step 1: logup remainder range
//     // Prover now returns ck as well
//     let (commit_step1, proof_step1, a_step1, t_step1) = prover.prove_step1_logup();
//     let result_logup_step1 =
//         verifier.verify_step1_logup(&commit_step1, &proof_step1, &a_step1, &t_step1);
//     assert!(
//         result_logup_step1,
//         "Step 1: Logup remainder range verification failed"
//     );

//     // Step 2: logup relu
//     let (commit_step2, proof_step2, a_step2, t_step2) = prover.prove_step2_logup();
//     let result_logup_step2 =
//         verifier.verify_step2_logup(&commit_step2, &proof_step2, &a_step2, &t_step2);
//     assert!(result_logup_step2, "Step 2: Logup relu verification failed");

//     println!("All steps verified successfully");
// }
