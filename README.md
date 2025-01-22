# sparse-differentiation-paper

TMLR submission **"Sparser, Better, Faster, Stronger: Efficient Automatic
Differentiation for Sparse Jacobians and Hessians"** 
on automatic sparse differentiation (ASD) and sparsity pattern detection.

## Code 

Our code is implemented in the [Julia programming language](https://julialang.org/downloads/).
We provide the full code and matching Julia environments needed to reproduce all experiments, examples, tables and figures in the paper.

### Installation

1. Clone or download this repository.
2. [Install Julia](https://julialang.org/downloads/). On Unix systems, this requires running
    ```bash
    curl -fsSL https://install.julialang.org | sh
    ```
3. Start a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) session by typing `julia` in your terminal
4. Run one of the experiments / examples listed below by typing `include("path/to/file.jl")` in your REPL, replacing the string with the correct path.

### Experiments

* Experiments on Jacobian sparsity detection (Section 5.1) and Jacobian computation (Section 5.2) can be found in [`/experiments/benchmark_brusselator/`](/experiments/benchmark_brusselator/):
    * `run_benchmarks.jl`: Run all benchmarks and save the results. This code also saves the Jacobian sparsity pattern figures (Section A.7, Figure 4).
    * `eval.jl`: Load the results and create benchmark tables (Tables 3 and 6).
    * `plot.jl`: Load the results and create the benchmark plot (Figure 2)
* Experiments on Hessian sparsity detection (Section 5.3) and Hessian computation (Section 5.4) can be found in [`/experiments/benchmark_acopf/`](/experiments/benchmark_acopf/):
    * `run_benchmarks.jl`: Run all benchmarks and save the results.
     This requires the installation of the [Rosetta OPF](https://github.com/lanl-ansi/rosetta-opf) implementations, as outlined in [`/experiments/benchmark_acopf/README.md`](/experiments/benchmark_acopf/README.md).
    * `eval.jl`: Load the results and save benchmark tables (Tables 4, 5, 7 and 8).
* The figures of colored Jacobian sparsity patterns of convolutional layers (Section A.6, Figure 3) are created by running [`/experiments/sparsity_patterns_convolution/conv.jl`](/experiments/sparsity_patterns_convolution/conv.jl)

### Examples

* The code demonstration from Section A.6 can be found in [`/examples/convolution/`](/examples/convolution/):
    * `pattern_detection.jl`: Code example from Listing 1
    * `jacobian_computation.jl`: Code example from Listing 2
