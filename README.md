# zkConv

## Overview

zkConv is a state-of-the-art zero-knowledge proof system designed for convolutional neural networks (CNNs). It leverages cutting-edge cryptographic techniques to ensure both efficiency and scalability, making it one of the fastest zero-knowledge proof systems for CNNs available today.

## Project Structure

The project is organized into the following directories:

- **`Cargo.lock` and `Cargo.toml`**: Rust package manager files to track dependencies and project metadata.
- **`benches`**: Contains benchmarking scripts to evaluate the performance of the proof system.
- **`crates`**: Includes modular libraries and components that form the building blocks of zkConv.
- **`dat`**: Stores real computational traces from the VGG16 model, providing authentic data for experiments and testing.
- **`src`**: The main source code of zkConv, including core logic and implementations.
- **`tests`**: Unit and integration tests to ensure correctness and reliability of the system.

## How to Run Benchmarks

All experimental data can be generated using `cargo bench`. This will run the benchmarking suite and output performance metrics for the proof system. To execute the benchmarks:

1. Ensure you have the Rust toolchain installed, preferably the nightly version (e.g., `rustc 1.86.0-nightly`).
2. Navigate to the project root directory.
3. Run the following command:
   ```sh
   cargo bench
   ```

The results will include detailed performance insights, including execution time and resource usage for various CNN proof scenarios.

## Installation and Usage

To use zkConv in your project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/zkConv.git
   ```
2. Navigate to the project directory:
   ```sh
   cd zkConv
   ```
3. Build the project:
   ```sh
   cargo build --release
   ```
4. Run examples or benchmarks as needed:
   ```sh
   cargo run --example <example_name>
   ```

## Contributing

We welcome contributions to zkConv! If you have ideas for improvements, please feel free to open an issue or submit a pull request. Follow the standard Rust community guidelines and ensure your code passes the tests and benchmarks.

## License

zkConv is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the license terms.

---

For more information or questions, please contact the project maintainers.
