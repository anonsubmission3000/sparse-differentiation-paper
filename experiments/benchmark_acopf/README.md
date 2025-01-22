# ACOPF comparison experiments

To get started, open a terminal and navigate to the `experiments/benchmark_acopf` subfolder.

## Installation

### Adding `rosetta-opf`

This experiment depends on the [Rosetta OPF](https://github.com/lanl-ansi/rosetta-opf) implementation of the AC-OPF benchmark.

For full reproducibility, clone the repository and use `rosetta-opf` commit `496d02671eb123368928b8e384e84f900db7cc3c`:

```bash
git clone git@github.com:lanl-ansi/rosetta-opf.git
cd rosetta-opf
git checkout 496d02671eb123368928b8e384e84f900db7cc3c
cd ..
```

### Adding the `pglib-opf` cases

Analogous to the installation of `rosetta-opf`, clone `pglib-opf` at commit `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`:

```bash
git clone git@github.com:power-grid-lib/pglib-opf.git
cd pglib-opf
git checkout dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3
cd ..
```

### Adding Julia dependencies

Now launch Julia 1.10 and install the required dependencies:

```julia
using Pkg; Pkg.instantiate()
```

## Running experiments

In the Julia REPL, you can re-run the experiments as follows:

```julia
include("run_benchmarks.jl")
```
