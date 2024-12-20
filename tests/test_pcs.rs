use ark_bls12_377::G1Affine;
use ark_ec::AffineRepr;
use ark_ed_on_bls12_381::EdwardsAffine;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_poly_commit::{
    data_structures::LinearCombination, hyrax::HyraxPC, LabeledPolynomial, PolynomialCommitment,
};
// include test_pcs_utils.rs
use ark_std::test_rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

// The test structure is largely taken from the multilinear_ligero module
// inside this crate

// ****************** types ******************

type Fq = <G1Affine as AffineRepr>::ScalarField;
type Hyrax377 = HyraxPC<G1Affine, DenseMultilinearExtension<Fq>>;

type Fr = <EdwardsAffine as AffineRepr>::ScalarField;
type Hyrax381 = HyraxPC<EdwardsAffine, DenseMultilinearExtension<Fr>>;

// ******** auxiliary test functions ********

fn rand_poly<Fr: PrimeField>(
    _: usize, // degree: unused
    num_vars: Option<usize>,
    rng: &mut ChaCha20Rng,
) -> DenseMultilinearExtension<Fr> {
    match num_vars {
        Some(n) => DenseMultilinearExtension::rand(n, rng),
        None => panic!("Must specify the number of variables"),
    }
}

fn constant_poly<Fr: PrimeField>(
    _: usize, // degree: unused
    num_vars: Option<usize>,
    rng: &mut ChaCha20Rng,
) -> DenseMultilinearExtension<Fr> {
    match num_vars {
        Some(0) => DenseMultilinearExtension::rand(0, rng),
        _ => panic!("Must specify the number of variables: 0"),
    }
}

fn rand_point<F: PrimeField>(num_vars: Option<usize>, rng: &mut ChaCha20Rng) -> Vec<F> {
    match num_vars {
        Some(n) => (0..n).map(|_| F::rand(rng)).collect(),
        None => panic!("Must specify the number of variables"),
    }
}

// ****************** tests ******************

#[test]
fn test_hyrax_construction() {
    // Desired number of variables (must be even!)
    let n = 8;

    let chacha = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();

    let pp = Hyrax381::setup(1, Some(n), chacha).unwrap();

    let (ck, vk) = Hyrax381::trim(&pp, 1, 1, None).unwrap();

    let l_poly = LabeledPolynomial::new(
        "test_poly".to_string(),
        rand_poly::<Fr>(0, Some(n), chacha),
        None,
        None,
    );

    let (c, rands) = Hyrax381::commit(&ck, &[l_poly.clone()], Some(chacha)).unwrap();

    let point: Vec<Fr> = rand_point(Some(n), chacha);
    let value = l_poly.evaluate(&point);

    // Dummy argument
    let mut test_sponge = test_sponge::<Fr>();

    let proof = Hyrax381::open(
        &ck,
        &[l_poly],
        &c,
        &point,
        &mut (test_sponge.clone()),
        &rands,
        Some(chacha),
    )
    .unwrap();

    assert!(Hyrax381::check(
        &vk,
        &c,
        &point,
        [value],
        &proof,
        &mut test_sponge,
        Some(chacha),
    )
    .unwrap());
}

#[test]
fn hyrax_single_poly_test() {
    single_poly_test::<_, _, Hyrax377, _>(
        Some(10),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    single_poly_test::<_, _, Hyrax381, _>(
        Some(10),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

#[test]
fn hyrax_constant_poly_test() {
    single_poly_test::<_, _, Hyrax377, _>(
        Some(0),
        constant_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    single_poly_test::<_, _, Hyrax381, _>(
        Some(0),
        constant_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

#[test]
fn hyrax_full_end_to_end_test() {
    full_end_to_end_test::<_, _, Hyrax377, _>(
        Some(8),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    full_end_to_end_test::<_, _, Hyrax381, _>(
        Some(10),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

#[test]
fn hyrax_single_equation_test() {
    single_equation_test::<_, _, Hyrax377, _>(
        Some(6),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    single_equation_test::<_, _, Hyrax381, _>(
        Some(6),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

#[test]
fn hyrax_two_equation_test() {
    two_equation_test::<_, _, Hyrax377, _>(
        Some(10),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    two_equation_test::<_, _, Hyrax381, _>(
        Some(10),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

#[test]
fn hyrax_full_end_to_end_equation_test() {
    full_end_to_end_equation_test::<_, _, Hyrax377, _>(
        Some(8),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fq>,
    )
    .expect("test failed for bls12-377");
    full_end_to_end_equation_test::<_, _, Hyrax381, _>(
        Some(8),
        rand_poly,
        rand_point,
        poseidon_sponge_for_test::<Fr>,
    )
    .expect("test failed for bls12-381");
}

use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_poly::Polynomial;
use ark_std::collections::{BTreeMap, BTreeSet};

/// `QuerySet` is the set of queries that are to be made to a set of labeled polynomials/equations
/// `p` that have previously been committed to. Each element of a `QuerySet` is a pair of
/// `(label, (point_label, point))`, where `label` is the label of a polynomial in `p`,
/// `point_label` is the label for the point (e.g., "beta"), and  and `point` is the location
/// that `p[label]` is to be queried at.
pub type QuerySet<T> = BTreeSet<(String, (String, T))>;

/// `Evaluations` is the result of querying a set of labeled polynomials or equations
/// `p` at a `QuerySet` `Q`. It maps each element of `Q` to the resulting evaluation.
/// That is, if `(label, query)` is an element of `Q`, then `evaluation.get((label, query))`
/// should equal `p[label].evaluate(query)`.
pub type Evaluations<T, F> = BTreeMap<(String, T), F>;

pub(crate) fn test_sponge<F: PrimeField>() -> PoseidonSponge<F> {
    use ark_crypto_primitives::sponge::{poseidon::PoseidonConfig, CryptographicSponge};
    use ark_std::test_rng;

    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut v = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        v.push(res);
    }
    let config = PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, v, 2, 1);
    PoseidonSponge::new(&config)
}

use ark_crypto_primitives::sponge::{poseidon::PoseidonConfig, CryptographicSponge};
use ark_std::rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

struct TestInfo<F: PrimeField, P: Polynomial<F>, S: CryptographicSponge> {
    num_iters: usize,
    max_degree: Option<usize>,
    supported_degree: Option<usize>,
    num_vars: Option<usize>,
    num_polynomials: usize,
    enforce_degree_bounds: bool,
    max_num_queries: usize,
    num_equations: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
}

pub fn bad_degree_bound_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let sponge = sponge();

    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let max_degree = 100;
    let pp = PC::setup(max_degree, None, rng)?;
    for _ in 0..10 {
        let supported_degree = Uniform::from(1..=max_degree).sample(rng);
        assert!(
            max_degree >= supported_degree,
            "max_degree < supported_degree"
        );

        let mut labels = Vec::new();
        let mut polynomials = Vec::new();
        let mut degree_bounds = Vec::new();

        for i in 0..10 {
            let label = format!("Test{}", i);
            labels.push(label.clone());
            let degree_bound = 1usize;
            let hiding_bound = Some(1);
            degree_bounds.push(degree_bound);

            polynomials.push(LabeledPolynomial::new(
                label,
                rand_poly(supported_degree, None, rng),
                Some(degree_bound),
                hiding_bound,
            ));
        }

        let supported_hiding_bound = polynomials
            .iter()
            .map(|p| p.hiding_bound().unwrap_or(0))
            .max()
            .unwrap_or(0);
        println!("supported degree: {:?}", supported_degree);
        println!("supported hiding bound: {:?}", supported_hiding_bound);
        let (ck, vk) = PC::trim(
            &pp,
            supported_degree,
            supported_hiding_bound,
            Some(degree_bounds.as_slice()),
        )?;
        println!("Trimmed");

        let (comms, rands) = PC::commit(&ck, &polynomials, Some(rng))?;

        let mut query_set = QuerySet::new();
        let mut values = Evaluations::new();
        let point = rand_point(None, rng);
        for (i, label) in labels.iter().enumerate() {
            query_set.insert((label.clone(), (format!("{}", i), point.clone())));
            let value = polynomials[i].evaluate(&point);
            values.insert((label.clone(), point.clone()), value);
        }
        println!("Generated query set");

        let proof = PC::batch_open(
            &ck,
            &polynomials,
            &comms,
            &query_set,
            &mut (sponge.clone()),
            &rands,
            Some(rng),
        )?;
        let result = PC::batch_check(
            &vk,
            &comms,
            &query_set,
            &values,
            &proof,
            &mut (sponge.clone()),
            rng,
        )?;
        assert!(result, "proof was incorrect, Query set: {:#?}", query_set);
    }

    Ok(())
}

fn test_template<F, P, PC, S>(info: TestInfo<F, P, S>) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let TestInfo {
        num_iters,
        max_degree,
        supported_degree,
        num_vars,
        num_polynomials,
        enforce_degree_bounds,
        max_num_queries,
        num_equations: _,
        rand_poly,
        rand_point,
        sponge,
    } = info;

    let sponge = sponge();

    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    // If testing multivariate polynomials, make the max degree lower
    let max_degree = match num_vars {
        Some(_) => max_degree.unwrap_or(Uniform::from(2..=10).sample(rng)),
        None => max_degree.unwrap_or(Uniform::from(2..=64).sample(rng)),
    };
    let pp = PC::setup(max_degree, num_vars, rng)?;

    for _ in 0..num_iters {
        let supported_degree =
            supported_degree.unwrap_or(Uniform::from(1..=max_degree).sample(rng));
        assert!(
            max_degree >= supported_degree,
            "max_degree < supported_degree"
        );
        let mut polynomials: Vec<LabeledPolynomial<F, P>> = Vec::new();
        let mut degree_bounds = if enforce_degree_bounds {
            Some(Vec::new())
        } else {
            None
        };

        let mut labels = Vec::new();
        println!("Sampled supported degree");

        // Generate polynomials
        let num_points_in_query_set = Uniform::from(1..=max_num_queries).sample(rng);
        for i in 0..num_polynomials {
            let label = format!("Test{}", i);
            labels.push(label.clone());
            let degree = Uniform::from(1..=supported_degree).sample(rng);
            let degree_bound = if let Some(degree_bounds) = &mut degree_bounds {
                let range = Uniform::from(degree..=supported_degree);
                let degree_bound = range.sample(rng);
                degree_bounds.push(degree_bound);
                Some(degree_bound)
            } else {
                None
            };

            let hiding_bound = if num_points_in_query_set >= degree {
                Some(degree)
            } else {
                Some(num_points_in_query_set)
            };

            polynomials.push(LabeledPolynomial::new(
                label,
                rand_poly(degree, num_vars, rng).into(),
                degree_bound,
                hiding_bound,
            ))
        }
        let supported_hiding_bound = polynomials
            .iter()
            .map(|p| p.hiding_bound().unwrap_or(0))
            .max()
            .unwrap_or(0);
        println!("supported degree: {:?}", supported_degree);
        println!("supported hiding bound: {:?}", supported_hiding_bound);
        println!("num_points_in_query_set: {:?}", num_points_in_query_set);
        let (ck, vk) = PC::trim(
            &pp,
            supported_degree,
            supported_hiding_bound,
            degree_bounds.as_ref().map(|s| s.as_slice()),
        )?;
        println!("Trimmed");

        let (comms, rands) = PC::commit(&ck, &polynomials, Some(rng))?;

        // Construct query set
        let mut query_set = QuerySet::new();
        let mut values = Evaluations::new();
        for _ in 0..num_points_in_query_set {
            let point = rand_point(num_vars, rng);
            for (i, label) in labels.iter().enumerate() {
                query_set.insert((label.clone(), (format!("{}", i), point.clone())));
                let value = polynomials[i].evaluate(&point);
                values.insert((label.clone(), point.clone()), value);
            }
        }
        println!("Generated query set");

        let proof = PC::batch_open(
            &ck,
            &polynomials,
            &comms,
            &query_set,
            &mut (sponge.clone()),
            &rands,
            Some(rng),
        )?;
        let result = PC::batch_check(
            &vk,
            &comms,
            &query_set,
            &values,
            &proof,
            &mut (sponge.clone()),
            rng,
        )?;
        if !result {
            println!(
                "Failed with {} polynomials, num_points_in_query_set: {:?}",
                num_polynomials, num_points_in_query_set
            );
            println!("Degree of polynomials:",);
            for poly in polynomials {
                println!("Degree: {:?}", poly.degree());
            }
        }
        assert!(result, "proof was incorrect, Query set: {:#?}", query_set);
    }

    Ok(())
}

fn equation_test_template<F, P, PC, S>(info: TestInfo<F, P, S>) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let TestInfo {
        num_iters,
        max_degree,
        supported_degree,
        num_vars,
        num_polynomials,
        enforce_degree_bounds,
        max_num_queries,
        num_equations,
        rand_poly,
        rand_point,
        sponge,
    } = info;

    let sponge = sponge();

    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    // If testing multivariate polynomials, make the max degree lower
    let max_degree = match num_vars {
        Some(_) => max_degree.unwrap_or(Uniform::from(2..=10).sample(rng)),
        None => max_degree.unwrap_or(Uniform::from(2..=64).sample(rng)),
    };
    let pp = PC::setup(max_degree, num_vars, rng)?;

    for _ in 0..num_iters {
        let supported_degree =
            supported_degree.unwrap_or(Uniform::from(1..=max_degree).sample(rng));
        assert!(
            max_degree >= supported_degree,
            "max_degree < supported_degree"
        );
        let mut polynomials = Vec::new();
        let mut degree_bounds = if enforce_degree_bounds {
            Some(Vec::new())
        } else {
            None
        };

        let mut labels = Vec::new();
        println!("Sampled supported degree");

        // Generate polynomials
        let num_points_in_query_set = Uniform::from(1..=max_num_queries).sample(rng);
        for i in 0..num_polynomials {
            let label = format!("Test{}", i);
            labels.push(label.clone());
            let degree = Uniform::from(1..=supported_degree).sample(rng);
            let degree_bound = if let Some(degree_bounds) = &mut degree_bounds {
                if rng.gen() {
                    let range = Uniform::from(degree..=supported_degree);
                    let degree_bound = range.sample(rng);
                    degree_bounds.push(degree_bound);
                    Some(degree_bound)
                } else {
                    None
                }
            } else {
                None
            };

            let hiding_bound = if num_points_in_query_set >= degree {
                Some(degree)
            } else {
                Some(num_points_in_query_set)
            };
            println!("Hiding bound: {:?}", hiding_bound);

            polynomials.push(LabeledPolynomial::new(
                label,
                rand_poly(degree, num_vars, rng),
                degree_bound,
                hiding_bound,
            ))
        }
        println!("supported degree: {:?}", supported_degree);
        println!("num_points_in_query_set: {:?}", num_points_in_query_set);
        println!("{:?}", degree_bounds);
        println!("{}", num_polynomials);
        println!("{}", enforce_degree_bounds);

        let (ck, vk) = PC::trim(
            &pp,
            supported_degree,
            supported_degree,
            degree_bounds.as_ref().map(|s| s.as_slice()),
        )?;
        println!("Trimmed");

        let (comms, rands) = PC::commit(&ck, &polynomials, Some(rng))?;

        // Let's construct our equations
        let mut linear_combinations = Vec::new();
        let mut query_set = QuerySet::new();
        let mut values = Evaluations::new();
        for i in 0..num_points_in_query_set {
            let point = rand_point(num_vars, rng);
            for j in 0..num_equations.unwrap() {
                let label = format!("query {} eqn {}", i, j);
                let mut lc = LinearCombination::empty(label.clone());

                let mut value = F::zero();
                let should_have_degree_bounds: bool = rng.gen();
                for (k, label) in labels.iter().enumerate() {
                    if should_have_degree_bounds {
                        value += &polynomials[k].evaluate(&point);
                        lc.push((F::one(), label.to_string().into()));
                        break;
                    } else {
                        let poly = &polynomials[k];
                        if poly.degree_bound().is_some() {
                            continue;
                        } else {
                            assert!(poly.degree_bound().is_none());
                            let coeff = F::rand(rng);
                            value += &(coeff * poly.evaluate(&point));
                            lc.push((coeff, label.to_string().into()));
                        }
                    }
                }
                values.insert((label.clone(), point.clone()), value);
                if !lc.is_empty() {
                    linear_combinations.push(lc);
                    // Insert query
                    query_set.insert((label.clone(), (format!("{}", i), point.clone())));
                }
            }
        }
        if linear_combinations.is_empty() {
            continue;
        }
        println!("Generated query set");
        println!("Linear combinations: {:?}", linear_combinations);

        let proof = PC::open_combinations(
            &ck,
            &linear_combinations,
            &polynomials,
            &comms,
            &query_set,
            &mut (sponge.clone()),
            &rands,
            Some(rng),
        )?;
        println!("Generated proof");
        let result = PC::check_combinations(
            &vk,
            &linear_combinations,
            &comms,
            &query_set,
            &values,
            &proof,
            &mut (sponge.clone()),
            rng,
        )?;
        if !result {
            println!(
                "Failed with {} polynomials, num_points_in_query_set: {:?}",
                num_polynomials, num_points_in_query_set
            );
            println!("Degree of polynomials:",);
            for poly in polynomials {
                println!("Degree: {:?}", poly.degree());
            }
        }
        assert!(
            result,
            "proof was incorrect, equations: {:#?}",
            linear_combinations
        );
    }

    Ok(())
}

pub fn single_poly_test<F, P, PC, S>(
    num_vars: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars,
        num_polynomials: 1,
        enforce_degree_bounds: false,
        max_num_queries: 1,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn linear_poly_degree_bound_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: Some(2),
        supported_degree: Some(1),
        num_vars: None,
        num_polynomials: 1,
        enforce_degree_bounds: true,
        max_num_queries: 1,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn single_poly_degree_bound_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars: None,
        num_polynomials: 1,
        enforce_degree_bounds: true,
        max_num_queries: 1,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn quadratic_poly_degree_bound_multiple_queries_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: Some(3),
        supported_degree: Some(2),
        num_vars: None,
        num_polynomials: 1,
        enforce_degree_bounds: true,
        max_num_queries: 2,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn single_poly_degree_bound_multiple_queries_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars: None,
        num_polynomials: 1,
        enforce_degree_bounds: true,
        max_num_queries: 2,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn two_polys_degree_bound_single_query_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars: None,
        num_polynomials: 2,
        enforce_degree_bounds: true,
        max_num_queries: 1,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn full_end_to_end_test<F, P, PC, S>(
    num_vars: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars,
        num_polynomials: 10,
        enforce_degree_bounds: true,
        max_num_queries: 5,
        num_equations: None,
        rand_poly,
        rand_point,
        sponge,
    };
    test_template::<F, P, PC, S>(info)
}

pub fn full_end_to_end_equation_test<F, P, PC, S>(
    num_vars: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars,
        num_polynomials: 10,
        enforce_degree_bounds: true,
        max_num_queries: 5,
        num_equations: Some(10),
        rand_poly,
        rand_point,
        sponge,
    };
    equation_test_template::<F, P, PC, S>(info)
}

pub fn single_equation_test<F, P, PC, S>(
    num_vars: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars,
        num_polynomials: 1,
        enforce_degree_bounds: false,
        max_num_queries: 1,
        num_equations: Some(1),
        rand_poly,
        rand_point,
        sponge,
    };
    equation_test_template::<F, P, PC, S>(info)
}

pub fn two_equation_test<F, P, PC, S>(
    num_vars: Option<usize>,
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars,
        num_polynomials: 2,
        enforce_degree_bounds: false,
        max_num_queries: 1,
        num_equations: Some(2),
        rand_poly,
        rand_point,
        sponge,
    };
    equation_test_template::<F, P, PC, S>(info)
}

pub fn two_equation_degree_bound_test<F, P, PC, S>(
    rand_poly: fn(usize, Option<usize>, &mut ChaCha20Rng) -> P,
    rand_point: fn(Option<usize>, &mut ChaCha20Rng) -> P::Point,
    sponge: fn() -> S,
) -> Result<(), PC::Error>
where
    F: PrimeField,
    P: Polynomial<F>,
    PC: PolynomialCommitment<F, P>,
    S: CryptographicSponge,
{
    let info = TestInfo {
        num_iters: 100,
        max_degree: None,
        supported_degree: None,
        num_vars: None,
        num_polynomials: 2,
        enforce_degree_bounds: true,
        max_num_queries: 1,
        num_equations: Some(2),
        rand_poly,
        rand_point,
        sponge,
    };
    equation_test_template::<F, P, PC, S>(info)
}

pub(crate) fn poseidon_sponge_for_test<F: PrimeField>() -> PoseidonSponge<F> {
    PoseidonSponge::new(&poseidon_parameters_for_test())
}

/// Generate default parameters for alpha = 17, state-size = 8
///
/// WARNING: This poseidon parameter is not secure. Please generate
/// your own parameters according the field you use.
pub(crate) fn poseidon_parameters_for_test<F: PrimeField>() -> PoseidonConfig<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut ark = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        ark.push(res);
    }
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}
