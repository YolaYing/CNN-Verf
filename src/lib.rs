pub mod conv;
pub mod conv_padding;
pub mod conv_padding_old;
pub mod maxpool;
pub mod relu;
pub mod relu_old;

use ark_bn254::{Bn254, Fr as ScalarField, FrConfig};
use ark_ec::pairing::Pairing;

pub type E = Bn254;
pub type F = <E as Pairing>::ScalarField;
