use ark_ec::pairing::Pairing;
use ark_std::{rand::Rng, test_rng, UniformRand};
use utils::rand_eval;
use zkconv::{E, F};

use pcs::{
    hyrax_kzg::hyrax_kzg_1::HyraxKzgPCS1,
    multilinear_kzg::data_structures::{
        MultilinearProverParam, MultilinearUniversalParams, MultilinearVerifierParam,
    },
    PolynomialCommitmentScheme,
};

// Helper function for testing a single instance of HyraxKzgPCS1
fn test_single_instance<R: Rng>(srs: &MultilinearUniversalParams<E>, eval: &Vec<F>, rng: &mut R) {
    // 计算多变量数量
    let num_vars = eval.len().ilog2();

    // 从SRS中生成Prover和Verifier参数
    let (ck, vk) = HyraxKzgPCS1::trim(srs);

    // 生成承诺
    let commit = HyraxKzgPCS1::commit(&ck, eval);

    // 随机生成点用于证明
    let point: Vec<_> = (0..num_vars).map(|_| F::rand(rng)).collect();

    // 打开证明并验证
    let (proof, value) = HyraxKzgPCS1::open(&ck, eval, &point);
    assert!(HyraxKzgPCS1::verify(&vk, &commit, &point, &proof, value));

    // 使用一个错误的值进行验证，确保验证失败
    let wrong_value = F::rand(rng);
    assert!(!HyraxKzgPCS1::verify(
        &vk,
        &commit,
        &point,
        &proof,
        wrong_value
    ));
}

#[test]
fn test_hyrax_kzg_pcs1() {
    // 初始化随机数生成器
    let mut rng = test_rng();

    // 测试的变量数量（2^10个评估点）
    let num_vars = 10;

    // 生成通用SRS
    let srs = HyraxKzgPCS1::<E>::gen_srs(&mut rng, num_vars);

    // 随机生成评估点
    let eval = rand_eval(num_vars, &mut rng);

    // 测试HyraxKzgPCS1实例
    test_single_instance(&srs, &eval, &mut rng);
}
