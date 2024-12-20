// to prove padding and rotation of convolution
// input:
// 1. original x without padding, x = c*n_x^2, c is the number of channels, n_x is the original size of the image
// 2. padded x, n_x_padded = n_x + 2*padding, x_padded = c*n_x_padded^2
// 3. calculated y, y = d*(len_x+len_w)
// 4. real y, y_real = d*n_y^2, n_y = n_x + 2*padding - len_w + 1
// to prove:
// 1. using permutation check to prove the padding process of x is correct
//     that is to prove the two set are equal:
//     the set of (x_padded[i], i) = the set of {(x[ci*w_in*w_in+(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)],i),when xi==0 or yi==0 or xi==padd_w-1 or yi==padd_w-1 -> +(0,i)}
//    to do that, we need to do the following steps:
//    1.1. calculate set 1: (x_padded[i], i) according to x_padded
//    1.2. calculate set 2: {(x[ci*w_in*w_in+(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)],i)+(0,not overed index) according to x
//    1.3. for each set, regard the first column as polynomial f, the second column as polynomial g
//    1.4. get a random number from verifier and combine f and g as h = f+g*random_number
//    1.5. prove using permuation check interface to prove the h from set 1 should be equal to the h from set 2
// 2. using permutation check to prove the rotation process of convolution is correct
//     that is to prove the two set are equal:
//     the set of calculated_y[co*(len_x+len_w)+(padd_w-1-xi)*padd_w+padd_w-1-yi],co*w_in*w_in+i) = the set of {(real_y[co*w_in*w_in+i],co*w_in*w_in+i)+(0,not covered index)}
//    to do that, we need to do the following steps(similar to step 1)
use ark_ff::PrimeField;
use ark_std::rand::Rng;
use merlin::Transcript;
use num_integer::Roots;
use poly_iop::perm_check::PermCheck;
use std::collections::VecDeque;

pub struct Prover<F: PrimeField> {
    x: Vec<F>,
    x_padded: Vec<F>,
    y: Vec<F>,
    y_real: Vec<F>,
    padding: usize,
    kernel_size: usize,
    input_channels: usize,
    output_channels: usize,
}

impl<F: PrimeField> Prover<F> {
    pub fn new(
        x: Vec<F>,
        x_padded: Vec<F>,
        y: Vec<F>,
        y_real: Vec<F>,
        padding: usize,
        kernel_size: usize,
        input_channels: usize,
        output_channels: usize,
    ) -> Self {
        Self {
            x,
            x_padded,
            y,
            y_real,
            padding,
            kernel_size,
            input_channels,
            output_channels,
        }
    }

    pub fn prove_padding<R: Rng>(
        &self,
        rng: &mut R,
        verifier_randomness: F,
    ) -> (VecDeque<Vec<F>>, Vec<F>, Vec<F>) {
        let (h_values_ori, h_values_padded) = self.generate_padding_h_values(verifier_randomness);

        let mut transcript = Transcript::new(b"PermCheck");
        let (proof, _, _) = PermCheck::prove(
            h_values_ori.clone(),
            h_values_padded.clone(),
            &mut transcript,
        );

        (proof, h_values_ori, h_values_padded)
    }

    pub fn prove_rotation<R: Rng>(
        &self,
        rng: &mut R,
        verifier_randomness: F,
    ) -> (VecDeque<Vec<F>>, Vec<F>, Vec<F>) {
        let (h_values_real, h_values_calculated) =
            self.generate_rotation_h_values(verifier_randomness);

        let mut transcript = Transcript::new(b"PermCheck");
        let (proof, _, _) = PermCheck::prove(
            h_values_calculated.clone(),
            h_values_real.clone(),
            &mut transcript,
        );

        (proof, h_values_calculated, h_values_real)
    }

    fn generate_padding_h_values(&self, verifier_randomness: F) -> (Vec<F>, Vec<F>) {
        let mut h_values_ori = Vec::new();
        let mut h_values_padded = Vec::new();
        let padd_w = (self.x_padded.len() / self.input_channels).sqrt();
        let w_in = (self.x.len() / self.input_channels).sqrt();

        // X_padded[ci*padd_w*padd_w+i]=x[ci*w_in*w_in+(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)]
        for ci in 0..self.input_channels {
            for i in 0..(padd_w * padd_w) {
                let xi = i / padd_w;
                let yi = i % padd_w;

                let original_val = if xi == 0 || yi == 0 || xi == padd_w - 1 || yi == padd_w - 1 {
                    F::zero()
                } else {
                    let xi_original = w_in - 1 - (xi - 1);
                    let yi_original = w_in - 1 - (yi - 1);
                    self.x[ci * w_in * w_in + xi_original * w_in + yi_original]
                };

                let padded_val = self.x_padded[ci * padd_w * padd_w + i];

                h_values_ori.push(
                    original_val + verifier_randomness * F::from((ci * padd_w * padd_w + i) as u64),
                );
                h_values_padded.push(
                    padded_val + verifier_randomness * F::from((ci * padd_w * padd_w + i) as u64),
                );
            }
        }

        (h_values_ori, h_values_padded)
    }

    fn generate_rotation_h_values(&self, verifier_randomness: F) -> (Vec<F>, Vec<F>) {
        let mut h_values_real = Vec::new();
        let mut h_values_calculated = Vec::new();
        let padd_w = (self.y_real.len() / self.output_channels).sqrt() + 2 * self.padding;
        let len_x = padd_w * padd_w;
        let len_w = padd_w * 3;
        let w_in = (self.y_real.len() / self.output_channels).sqrt();

        let mut covered_indices = std::collections::HashSet::new();

        for co in 0..self.output_channels {
            for i in 0..(w_in * w_in) {
                let xi = i / w_in;
                let yi = i % w_in;

                let real_val = self.y_real[co * (w_in * w_in) + i];
                h_values_real.push(
                    real_val + verifier_randomness * F::from((co * (w_in * w_in) + i) as u64),
                );

                let calculated_index =
                    co * (len_x + len_w) + (padd_w - 1 - xi) * padd_w + (padd_w - 1 - yi);

                covered_indices.insert(calculated_index);

                let calculated_val = self.y[calculated_index];
                h_values_calculated.push(
                    calculated_val + verifier_randomness * F::from((co * (w_in * w_in) + i) as u64),
                );
            }
        }

        // Add uncovered indices of calculated_y
        for i in 0..self.y.len() {
            if !covered_indices.contains(&i) {
                // let calculated_val = self.y[i];
                h_values_calculated.push(F::zero() + verifier_randomness * F::from(i as u64));
            }
        }

        for i in (self.y_real.len())..(self.y.len()) {
            h_values_real.push(F::zero() + verifier_randomness * F::from(i as u64));
        }

        if h_values_real == h_values_calculated {
            println!("h_values_real==h_values_calculated");
        } else {
            println!("h_values_real!=h_values_calculated");
        }

        (h_values_real, h_values_calculated)
    }
}
