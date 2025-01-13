use arithmetic::multilinear_poly::evaluate_on_point;
use ark_ff::PrimeField;
use merlin::Transcript;
use poly_iop::perm_check::PermCheck;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::num::ParseIntError;

pub struct Verifier<F: PrimeField> {
    num_vars: usize,         // Number of variables
    _marker: PhantomData<F>, // Marker to tie the type parameter F to the struct
}

impl<F: PrimeField> Verifier<F> {
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            _marker: PhantomData,
        }
    }

    /// Verifies the permutation check proof for two sets
    pub fn verify(
        &self,
        h_values_set1: Vec<F>,
        h_values_set2: Vec<F>,
        mut proof: VecDeque<Vec<F>>,
    ) -> bool {
        let mut transcript = Transcript::new(b"PermCheck");

        let next_power_of_two = h_values_set1.len().next_power_of_two() >> 1;

        let (challenges, values) = PermCheck::verify(
            next_power_of_two.trailing_zeros() as usize,
            &mut transcript,
            &mut proof,
        );

        // Check if the evaluated points match the expected values
        evaluate_on_point(&h_values_set1, &challenges[0]) == values[0]
            && evaluate_on_point(&h_values_set2, &challenges[1]) == values[1]
    }
}
