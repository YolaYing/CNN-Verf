//! Proving Padding and Rotation of Convolution
//!
//! ### Inputs:
//! 1. Original `x` without padding:
//!    - `x = c * n_x^2`
//!    - `c`: Number of channels
//!    - `n_x`: Original size of the image
//! 2. Padded `x`:
//!    - `n_x_padded = n_x + 2 * padding`
//!    - `x_padded = c.next_power_of_two * (n_x_padded^2).next_power_of_two`
//! 3. Calculated `Y` (rotated `y`):
//!    - `y = d * (len_x + len_w)`
//! 4. Real `y`:
//!    - `y_real = d * n_y^2`
//!    - `n_y = n_x + 2 * padding - len_w + 1`
//! 5. Updated comparison:
//!    - `P`: Calculated `Y = real_y || P`
//!    - `P`: All uncovered indices in calculated `Y` compared to real `y`
//!
//! ### To Prove:
//! #### 1. Padding Process:
//! - Use permutation check to prove the correctness of the padding process of `x`.
//! - Prove the equality of two sets:
//!   1. `(x_padded[i], i)` from `x_padded`
//!   2. `{(x[ci * w_in^2 + (w_in - 1 - (xi - 1)) * w_in + w_in - 1 - (yi - 1)], i)} + (0, i)`
//!      when `xi == 0` or `yi == 0` or `xi == padd_w - 1` or `yi == padd_w - 1`.
//!
//! #### Steps for Padding Proof:
//! 1. Calculate set 1: `(x_padded[i], i)`.
//! 2. Calculate set 2: `{(x[ci * w_in^2 + (w_in - 1 - (xi - 1)) * w_in + w_in - 1 - (yi - 1)], i)} + (0, uncovered index)`.
//! 3. For each set, regard the first column as polynomial `f` and the second column as polynomial `g`.
//! 4. Combine `f` and `g` with a random number from the verifier: `h = f + g * random_number`.
//! 5. Use the permutation check interface to prove `h` from set 1 equals `h` from set 2.
//!
//! #### 2. Rotation Process:
//! - Use permutation check to prove `calculated Y = real y || P`.
//! - Prove two subsets:
//!   1. Subset for `P`:
//!      - Prove `(calculated_y[co * (len_x + len_w) + p_pos[i]], co * w_in^2 + i)` equals `(P[co * w_in^2 + i], co * w_in^2 + i)`.
//!   2. Subset for `real_y`:
//!      - Prove `(calculated_y[co * (len_x + len_w) + (padd_w - 1 - xi) * padd_w + padd_w - 1 - yi], co * w_in^2 + i)` equals `(real_y[co * w_in^2 + i], co * w_in^2 + i)`.
//!
//! #### Steps for Rotation Proof:
//! 1. Calculate set 1: `(real_y[co * w_in^2 + i], co * w_in^2 + i)`.
//! 2. Calculate set 2: `(P[co * w_in^2 + i], co * w_in^2 + i)`.
//! 3. Calculate set 3: `(calculated_y[co * (len_x + len_w) + (padd_w - 1 - xi) * padd_w + padd_w - 1 - yi], co * w_in^2 + i)`.
//! 4. Calculate set 4: `(calculated_y[co * (len_x + len_w) + p_pos[i]], co * w_in^2 + i)`.
//! 5. For each set, regard the first column as polynomial `f` and the second column as polynomial `g`.
//! 6. Combine `f` and `g` with random number `r1`: `h = f + g * r1`. Obtain `h1, h2, h3, h4` from sets 1, 2, 3, 4.
//! 7. Combine `h1` and `h2` with random number `r2`: `real_y_concat_p = r2 * h1 + (1 - r2) * h2`.
//!    Combine `h3` and `h4` with `r2`: `calculated_y_concat = r2 * h3 + (1 - r2) * h4`.
//! 8. Use the permutation check interface to prove `real_y_concat_p` equals `calculated_y_concat`.

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
    p: Vec<F>,
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
        p: Vec<F>,
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
            p,
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
        verifier_randomness: Vec<F>,
    ) -> (VecDeque<Vec<F>>, Vec<F>, Vec<F>) {
        let (h_values_real_y_concate_p, h_values_calculated) =
            self.generate_rotation_h_values(verifier_randomness);

        let mut transcript = Transcript::new(b"PermCheck");
        let (proof, _, _) = PermCheck::prove(
            h_values_calculated.clone(),
            h_values_real_y_concate_p.clone(),
            &mut transcript,
        );

        (proof, h_values_calculated, h_values_real_y_concate_p)
    }

    fn generate_padding_h_values(&self, verifier_randomness: F) -> (Vec<F>, Vec<F>) {
        let mut h_values_ori = Vec::new();
        let mut h_values_padded = Vec::new();
        // let padd_w = (self.x_padded.len() / self.input_channels).sqrt();
        // w_in here is the original width of the image, x is the original image without padding, and input_channels is the number of input channels without padding
        // e.g. x = c * n_x^2, input_channels = c, w_in = n_x
        let w_in = (self.x.len() / self.input_channels).sqrt();
        // padd_w is the width of the image after padding
        let padd_w = w_in + 2 * self.padding;
        // PADD_channel is the number of input channels after padding(in VGG16, only the first layer needs input channel padding)
        let PADD_channel = self.input_channels.next_power_of_two();
        // PADD_X is the number of elements in each channel after padding
        // e.g original x: (3,32,32) -> PADD_X = ((32+2)^2).next_power_of_two = 2048
        let PADD_X = self.x_padded.len() / PADD_channel;

        // X_padded[ci*padd_w*padd_w+i]=x[ci*w_in*w_in+(w_in-1-(xi-1))*w_in+w_in-1-(yi-1)]
        for ci in 0..PADD_channel {
            for i in 0..(padd_w * padd_w) {
                let xi = i / padd_w;
                let yi = i % padd_w;

                if ci >= self.input_channels {
                    h_values_ori
                        .push(F::zero() + verifier_randomness * F::from((ci * PADD_X + i) as u64));

                    h_values_padded.push(
                        self.x_padded[ci * PADD_X + i]
                            + verifier_randomness * F::from((ci * PADD_X + i) as u64),
                    );
                    continue;
                }

                let original_val = if xi == 0 || yi == 0 || xi == padd_w - 1 || yi == padd_w - 1 {
                    F::zero()
                } else {
                    //     let xi_original = w_in - 1 - (xi - 1);
                    //     let yi_original = w_in - 1 - (yi - 1);
                    //     self.x[ci * w_in * w_in + xi_original * w_in + yi_original]
                    //
                    self.x[ci * w_in * w_in + (w_in - 1 - (xi - 1)) * w_in + w_in - 1 - (yi - 1)]
                };

                let padded_val = self.x_padded[ci * PADD_X + i];

                h_values_ori
                    .push(original_val + verifier_randomness * F::from((ci * PADD_X + i) as u64));
                h_values_padded
                    .push(padded_val + verifier_randomness * F::from((ci * PADD_X + i) as u64));
            }
            if padd_w * padd_w < PADD_X {
                for i in padd_w * padd_w..PADD_X {
                    h_values_ori
                        .push(F::zero() + verifier_randomness * F::from((ci * PADD_X + i) as u64));
                    h_values_padded.push(
                        self.x_padded[ci * PADD_X + i]
                            + verifier_randomness * F::from((ci * PADD_X + i) as u64),
                    );
                }
            }
        }

        (h_values_ori, h_values_padded)
    }

    fn generate_rotation_h_values(&self, verifier_randomness: Vec<F>) -> (Vec<F>, Vec<F>) {
        let mut h_values_real_y = Vec::new();
        let mut h_values_calculated_pair_with_real_y = Vec::new();
        let mut h_values_p = Vec::new();
        let mut h_values_calculated_pair_with_p = Vec::new();
        let mut h_values_real_y_concate_p = Vec::new();
        let mut h_values_calculated_y_concate = Vec::new();
        let padd_w = (self.x.len() / self.input_channels).sqrt() + 2 * self.padding;
        let len_x = padd_w * padd_w;
        let len_w = padd_w * 3;
        let PADD_Y = self.y_real.len() / self.output_channels;
        let w_in = (self.x.len() / self.input_channels).sqrt();

        // visit=[0 for i in range(PADD_Y)]
        let mut visited = vec![false; PADD_Y];

        for co in 0..self.output_channels {
            for i in 0..(w_in * w_in) {
                let xi = i / w_in;
                let yi = i % w_in;

                let real_val = self.y_real[co * PADD_Y + i];
                h_values_real_y
                    .push(real_val + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64));

                let calculated_index = co * PADD_Y + (padd_w - 1 - xi) * padd_w + (padd_w - 1 - yi);

                visited[(padd_w - 1 - xi) * padd_w + padd_w - 1 - yi] = true;

                let calculated_val = self.y[calculated_index];

                h_values_calculated_pair_with_real_y.push(
                    calculated_val + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64),
                );
            }
        }

        // calculate p_pos
        let mut p_pos = Vec::new();
        for i in 0..PADD_Y {
            if i < len_x + len_w {
                if !visited[i] {
                    p_pos.push(i);
                }
            } else {
                break;
            }
        }

        for co in 0..self.output_channels {
            for i in 0..p_pos.len() {
                let p_value = self.p[co * PADD_Y + i];
                h_values_p
                    .push(p_value + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64));

                let calculated_val = self.y[co * PADD_Y + p_pos[i]];
                h_values_calculated_pair_with_p.push(
                    calculated_val + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64),
                );
            }
            if p_pos.len() < PADD_Y {
                for i in p_pos.len()..PADD_Y {
                    h_values_p.push(
                        self.p[co * PADD_Y + i]
                            + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64),
                    );
                    h_values_calculated_pair_with_p.push(
                        F::zero() + verifier_randomness[0] * F::from((co * PADD_Y + i) as u64),
                    );
                }
            }
        }

        // compute h_values_real_y_concate_p and h_values_calculated_y_concate
        for i in 0..h_values_real_y.len() {
            h_values_real_y_concate_p.push(
                h_values_real_y[i] * verifier_randomness[1]
                    + h_values_p[i] * (F::one() - verifier_randomness[1]),
            );
            h_values_calculated_y_concate.push(
                h_values_calculated_pair_with_real_y[i] * verifier_randomness[1]
                    + h_values_calculated_pair_with_p[i] * (F::one() - verifier_randomness[1]),
            );
        }

        (h_values_real_y_concate_p, h_values_calculated_y_concate)
    }
}
