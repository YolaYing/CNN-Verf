use ark_ff::{Field, UniformRand};
use ark_poly::DenseMultilinearExtension;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    rc::Rc,
    test_rng,
};
use zkconv::conv::prover::Prover;
use zkconv::conv::verifier::{Verifier, VerifierMessage};
use zkconv::{E, F};

fn parse_dimensions(dim_str: &str) -> Vec<usize> {
    dim_str
        .trim_matches(&['(', ')'][..]) // Remove parentheses
        .split(',')
        .map(|v| v.trim().parse::<usize>().expect("Invalid dimension"))
        .collect()
}

fn parse_line(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }
    let parts: Vec<&str> = line.splitn(2, '[').collect();
    if parts.len() != 2 {
        panic!(
            "Invalid format: expected dimensions and data for '{}'",
            prefix
        );
    }

    // Parse dimensions from the part before '['
    let dim_part = parts[0]
        .split(':')
        .nth(1)
        .expect("Missing dimensions")
        .trim();
    let dims = parse_dimensions(dim_part);

    // Parse data from the part inside '[...]'
    let data_part = parts[1].trim_end_matches(']').trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (dims, values)
}

fn parse_line_with_shapes(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }

    // 提取括号中的两个形状以及数据部分
    let parts: Vec<&str> = line.split("padded shape:").collect();
    if parts.len() != 2 {
        panic!("Invalid format: expected padded shape in '{}'", line);
    }

    // 提取有用形状（useful shape）和填充形状（padded shape）
    let useful_shape = parts[0]
        .split("useful shape:")
        .nth(1)
        .expect("Missing useful shape")
        .trim();
    let padded_shape = parts[1]
        .split('[')
        .nth(0)
        .expect("Missing padded shape")
        .trim();

    // 解析维度
    let useful_dims = parse_dimensions(useful_shape);
    let padded_dims = parse_dimensions(padded_shape);

    // 确保维度一致
    if useful_dims.len() != padded_dims.len() {
        panic!("Mismatch between useful and padded dimensions");
    }

    // 提取数据部分
    let data_part = parts[1]
        .split('[')
        .nth(1)
        .expect("Missing data section")
        .trim_end_matches(']')
        .trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (padded_dims, values)
}

fn read_and_prepare_data<P: AsRef<std::path::Path>>(
    file_path: P,
) -> io::Result<(Vec<F>, Vec<F>, Vec<F>, usize, usize, usize, usize, usize)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Parse "rot pad x"
    let rot_pad_x_line = lines
        .nth(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot pad x line"))??;

    let (rot_pad_x_dims, rot_pad_x_values) = parse_line(&rot_pad_x_line, "rot pad x:");

    // Parse "weight W"
    let weight_w_line = lines
        .nth(0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing weight W line"))??;
    let (weight_w_dims, weight_w_values) = parse_line(&weight_w_line, "weight W:");

    // Parse "rot y"
    let rot_y_line = lines
        .nth(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot y line"))??;
    // let (_, rot_y_values) = parse_line(&rot_y_line, "rot y:");
    let (rot_y_dims, rot_y_values) = parse_line_with_shapes(&rot_y_line, "rot y:"); // 调用新的解析方法

    // Extract variable dimensions
    let input_channels = rot_pad_x_dims[0].next_power_of_two().trailing_zeros() as usize;
    let input_image = rot_pad_x_dims[1].next_power_of_two().trailing_zeros() as usize;
    let output_channels = weight_w_dims[0].next_power_of_two().trailing_zeros() as usize;
    let kernel_size = weight_w_dims[2].next_power_of_two().trailing_zeros() as usize;
    let output_data = rot_y_dims[1].next_power_of_two().trailing_zeros() as usize;

    // check if rot_pad_x_values.len() == 2^(input_channels + input_image)
    if rot_pad_x_values.len() != 1 << (input_channels + input_image) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid number of rot pad x values",
        ));
    }
    // check if weight_w_values.len() == 2^(output_channels + input_channels + kernel_size)
    if weight_w_values.len() != 1 << (output_channels + input_channels + kernel_size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid number of weight W values",
        ));
    }
    // check if rot_y_values.len() == 2^(output_channels + output_data)
    println!(
        "output_channels: {}, output_data: {}",
        output_channels, output_data
    );
    if rot_y_values.len() != 1 << (output_channels + output_data) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid number of rot y values",
        ));
    }

    println!(
        "input_channels: {}, input_image: {}, output_channels: {}, kernel_size: {}, output_data: {}",
        input_channels, input_image, output_channels, kernel_size, output_data
    );

    Ok((
        rot_pad_x_values,
        weight_w_values,
        rot_y_values,
        output_channels,
        output_data,
        input_channels,
        input_image,
        kernel_size,
    ))
}

#[test]
fn test_conv_real_data() {
    let file_path = "./dat/dat/conv_layer_29.txt";

    // Step 1: prepare data
    // for zk input X: e.g., 4(3->2^2)*2048(34*34->2^11), using 2 variables representing input channel, 11 variables indexing image
    // for zk input W: e.g., 64*4(3->2^2)*128(3*34->2^7), using 6 variables representing output channel, 2 variables representing input channel, 7 variables indexing kernel
    // for zk output Y: e.g., 64*2048(34*37 -> 2^11), using 6 variables representing output channel, 11 variables indexing output data
    let (x, w, y, num_vars_j, num_vars_s, num_vars_i, num_vars_a, num_vars_b) =
        read_and_prepare_data(file_path).expect("Failed to read convolution data");

    // Step 2: calculate the number of variables
    // These values are now dynamically calculated and read from the file

    // Create polynomials
    let x_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_i + num_vars_a, x);
    let w_poly =
        DenseMultilinearExtension::from_evaluations_vec(num_vars_i + num_vars_j + num_vars_b, w);
    let y_poly = DenseMultilinearExtension::from_evaluations_vec(num_vars_j + num_vars_s, y);

    let prover = Prover::new(
        Rc::new(y_poly),
        Rc::new(w_poly),
        Rc::new(x_poly),
        num_vars_j,
        num_vars_s,
        num_vars_i,
        num_vars_a,
        num_vars_b,
    );

    let verifier = Verifier::new(num_vars_j, num_vars_s, num_vars_i, num_vars_a, num_vars_b);

    // Step 3: mock verifier message
    let mut rng = test_rng();
    let r1_values: Vec<F> = (0..num_vars_j).map(|_| F::rand(&mut rng)).collect();
    let r = F::rand(&mut rng);
    let verifier_msg = VerifierMessage { r1_values, r };

    // Step 4: Prover generates proof
    let (
        proof_s,
        proof_f,
        proof_g,
        asserted_s,
        asserted_f,
        asserted_g,
        poly_info_s,
        poly_info_f,
        poly_info_g,
    ) = prover.prove(&mut rng, verifier_msg);

    // Step 5: Verifier verifies proof
    let result = verifier.verify(
        &proof_s,
        &proof_f,
        &proof_g,
        asserted_s,
        asserted_f,
        asserted_g,
        &poly_info_s,
        &poly_info_f,
        &poly_info_g,
    );

    assert!(result, "Verification failed");
    println!("Convolution layer verification with real data passed.");
}
