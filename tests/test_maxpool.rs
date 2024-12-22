use ark_ff::Zero;
use ark_poly::DenseMultilinearExtension;
use ark_std::rand::Rng;
use ark_std::rc::Rc;
use ark_std::test_rng;
use zkconv::{
    maxpool::{prover::Prover, verifier::Verifier},
    F,
};

// #[test]
// fn test_maxpool_proof() {
//     let mut rng = test_rng();

//     let num_vars_y1 = 4; // y1 has variables (b1, b2, b3, b4)
//     let num_vars_y2 = 2; // y2 has variables (b3, b4)

//     let size_y1 = 1 << num_vars_y1; // Total evaluations for y1
//     let size_y2 = 1 << num_vars_y2; // Total evaluations for y2

//     // Generate random evaluations for y1
//     let mut y1_values = Vec::with_capacity(size_y1);
//     for _ in 0..size_y1 {
//         y1_values.push(F::from(rng.gen_range(0u32..100)));
//     }

//     // Generate y2 as the maximum of y1 slices for each (b3, b4)
//     let mut y2_values = Vec::with_capacity(size_y2);
//     for sub_i in 0..size_y2 {
//         let i0 = sub_i; // (b1, b2) = (0, 0)
//         let i1 = 4 + sub_i; // (b1, b2) = (0, 1)
//         let i2 = 8 + sub_i; // (b1, b2) = (1, 0)
//         let i3 = 12 + sub_i; // (b1, b2) = (1, 1)
//         let max_val = y1_values[i0]
//             .max(y1_values[i1])
//             .max(y1_values[i2])
//             .max(y1_values[i3]);
//         y2_values.push(max_val);
//     }

//     let y1 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//         num_vars_y1,
//         y1_values,
//     ));
//     let y2 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
//         num_vars_y2,
//         y2_values,
//     ));

//     // Prover setup
//     let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);

//     // Prove using sumcheck
//     let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut rng);

//     // Prove inequalities using logup
//     let (commit, logup_proof, y2_evaluations, max_y1_evaluations) = prover.prove_inequalities();

//     // Verifier setup
//     let verifier = Verifier::new(num_vars_y2);

//     // Verify sumcheck proof
//     assert!(
//         verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info),
//         "Sumcheck verification failed"
//     );

//     // Verify logup proof
//     assert!(
//         verifier.verify_inequalities(&commit, &logup_proof, &y2_evaluations, &max_y1_evaluations),
//         "Logup verification failed"
//     );
// }

#[test]
fn test_maxpool_realnum_proof() {
    let mut rng = test_rng();

    let num_vars_y1 = 16; // y1 has 16 variables
    let num_vars_y2 = 14; // y2 has 14 variables after max operation

    let size_y1 = 1 << num_vars_y1; // Total evaluations for y1
    let size_y2 = 1 << num_vars_y2; // Total evaluations for y2

    // Generate random evaluations for y1
    let mut y1_values = Vec::with_capacity(size_y1);
    for _ in 0..size_y1 {
        y1_values.push(F::from(rng.gen_range(0u32..100)));
    }

    // Generate y2 as the maximum of y1 slices for each combination of the reduced variables
    let stride_b1b2 = 1 << (num_vars_y1 - num_vars_y2); // Number of evaluations for each (b3, ..., b16)
    let mut y2_values = Vec::with_capacity(size_y2);

    for sub_i in 0..size_y2 {
        let mut max_val = F::zero();
        for i in 0..stride_b1b2 {
            let index = i * size_y2 + sub_i; // Compute the index for this slice
            max_val = max_val.max(y1_values[index]);
        }
        y2_values.push(max_val);
    }

    let y1 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars_y1,
        y1_values,
    ));
    let y2 = Rc::new(DenseMultilinearExtension::from_evaluations_vec(
        num_vars_y2,
        y2_values,
    ));

    // Prover setup
    let prover = Prover::new(y1.clone(), y2.clone(), num_vars_y1, num_vars_y2);

    // Prove using sumcheck
    let (sumcheck_proof, asserted_sum, poly_info) = prover.prove_sumcheck(&mut rng);

    // Generate processed inequalities (excluded from prover time measurement)
    let (expanded_y2, combined_y1, a, range, commit, pk, ck) = prover.process_inequalities();

    // Prove inequalities using logup
    // let (commit, logup_proof, a, range) = prover.prove_inequalities();
    // Prove inequalities using logup
    let (commit_logup, logup_proof, a_proof, range_proof) =
        prover.prove_inequalities(&a, &range, &pk, commit.clone());

    // Verifier setup
    let verifier = Verifier::new(num_vars_y1);

    // Verify sumcheck proof
    assert!(
        verifier.verify_sumcheck(&sumcheck_proof, asserted_sum, &poly_info),
        "Sumcheck verification failed"
    );

    // Verify logup proof
    // assert!(
    //     verifier.verify_inequalities(&commit, &logup_proof, &a, &range),
    //     "Logup verification failed"
    // );
    assert!(
        verifier.verify_inequalities(&commit_logup, &logup_proof, &a_proof, &range_proof, &ck),
        "Logup verification failed"
    );
}
