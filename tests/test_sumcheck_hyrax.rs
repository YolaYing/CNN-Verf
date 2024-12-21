// #[test]
// fn test_with_hyrax_integration() {
//     let mut rng = ark_std::test_rng();
//     let nv = 8;

//     // 初始化 Hyrax 密钥
//     let (ck, vk) = HyraxPCS::setup(nv, Some(nv), &mut rng).unwrap();

//     // 构建随机多项式
//     let (poly, asserted_sum) = random_list_of_products::<F, _>(nv, (3, 4), 5, &mut rng);
//     let poly_info = poly.info();

//     // 初始化 Prover 和 Verifier
//     let mut prover_state = IPForMLSumcheck::prover_init_with_commitments(&poly, &ck);
//     let mut verifier_state = IPForMLSumcheck::verifier_init(&poly_info, Some(vk));

//     // 执行交互协议
//     for _ in 0..poly.num_variables {
//         let prover_msg = IPForMLSumcheck::prove_round(&mut prover_state, None);
//         let verifier_msg = IPForMLSumcheck::verify_round(
//             prover_msg,
//             &mut verifier_state,
//             &verifier_state.vk.as_ref().unwrap(),
//             &mut rng,
//         );
//         assert!(verifier_msg.is_some(), "Verifier message is missing.");
//     }

//     let subclaim = IPForMLSumcheck::check_and_generate_subclaim(verifier_state, asserted_sum)
//         .expect("Subclaim generation failed.");

//     assert!(
//         poly.evaluate(&subclaim.point) == subclaim.expected_evaluation,
//         "Verification failed."
//     );
// }
