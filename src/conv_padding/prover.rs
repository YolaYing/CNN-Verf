// to prove padding and rotation of convolution
// input:
// 1. original x without padding, x = c*n_x^2, c is the number of channels, n_x is the original size of the image
// 2. padded x, n_x_padded = n_x + 2*padding, x_padded = c*n_x_padded^2
// 3. calculated Y(rot y), y = d*(len_x+len_w)
// 4. real y, y_real = d*n_y^2, n_y = n_x + 2*padding - len_w + 1
// 5. (update compare to old version)P, calculated Y = real y || P, P is all the uncovered indices in calculated Y compared to real y
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
// 2. (update compare to old version)using permutation check to prove the rotation process of convolution is correct
//     that is to prove calculated Y = real y || P
//      for P is the subset of calculated Y, that is to prove the two set are equal:
//       the set of (calculated_y[co*(len_x+len_w)+p_pos[i]],co*w_in*w_in+i) = the set of (P[co*w_in*w_in+i],co*w_in*w_in+i)
//       where the p_pos is calculated from the uncoverd index of [co*(len_x+len_w)+(padd_w-1-xi)*padd_w+padd_w-1-yi]
//      for real y is the subset of calculated Y, that is to prove the two set are equal:
//       the set of (calculated_y[co*(len_x+len_w)+(padd_w-1-xi)*padd_w+padd_w-1-yi],co*w_in*w_in+i) = the set of (real_y[co*w_in*w_in+i],co*w_in*w_in+i)
//    to do that, we need to do the following steps:
//    1.1.  calculate set 1: (real_y[co*w_in*w_in+i],co*w_in*w_in+i) according to real_y
//    1.2.  calculate set 2: (P[co*w_in*w_in+i],co*w_in*w_in+i) according to P
//    1.3.  calculate set 3: (calculated_y[co*(len_x+len_w)+(padd_w-1-xi)*padd_w+padd_w-1-yi],co*w_in*w_in+i) according to calculated y
//    1.4.  calculate set 4: (calculated_y[co*(len_x+len_w)+p_pos[i]],co*w_in*w_in+i) according to calculated y
//    1.5.  for each set, regard the first column as polynomial f, the second column as polynomial g
//    1.6.  get a random number r1 from verifier and combine f and g as h = f+g*r1, and we can get h1,h2,h3,h4 from set 1,2,3,4
//    1.7.  get a random number r2 from verifier and:
//           combine h1 and h2 as real_y_concate_p = r2 * h1 + (1-r2) * h2
//           combine h3 and h4 as calculated_y_concate = r2 * h3 + (1-r2) * h4
//    1.8.  prove using permuation check interface to prove real_y_concate_p should be equal to calculated_y_concate

use crate::F;
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
        let w_in = (self.x.len() / self.input_channels).sqrt();
        let padd_w = w_in + 2 * self.padding;
        let PADD_channel = self.input_channels.next_power_of_two();
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
