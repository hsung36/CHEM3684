# CHEM3684 (montecarlo)

[![GitHub Actions Build Status](https://github.com/hsung36/CHEM3684/workflows/CI/badge.svg)](https://github.com/hsung36/CHEM3684/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/hsung36/CHEM3684/branch/main/graph/badge.svg)](https://codecov.io/gh/hsung36/CHEM3684/branch/main)

A Python package for performing Monte Carlo simulations on Ising models, including tools for bitstring manipulation and energy calculations.
## Overview

This package provides tools to simulate and analyze Ising models using Monte Carlo methods. It allows users to define lattice structures, compute energies of spin configurations, run Metropolis Monte Carlo simulations, and calculate various thermodynamic properties.

## Key Features

* **BitString Representation:** Efficiently represents and manipulates spin configurations.
* **Ising Hamiltonian:** Defines the Ising model on a given graph (lattice) and calculates energies.
* **Metropolis Monte Carlo:** Implements the Metropolis algorithm for simulating equilibrium states.
* **Thermodynamic Averages:** Computes average energy, magnetization, heat capacity, and magnetic susceptibility.
* **(Optional) Exact Enumeration:** For small systems, allows exact calculation of thermodynamic properties by enumerating all possible states.

## Installation

Currently, you can install this package directly from the GitHub repository:

```bash
pip install git+[https://github.com/hsung36/CHEM3684.git](https://github.com/hsung36/CHEM3684.git)
```


For developers, to install in editable mode (allowing you to modify the code and see changes immediately):

```bash
git clone [https://github.com/hsung36/CHEM3684.git](https://github.com/hsung36/CHEM3684.git)
cd CHEM3684
pip install -e .
```


Basic Usage
Here's a quick example of how to use the montecarlo components within the CHEM3684 package:

# Import necessary classes
from montecarlo.ising import IsingHamiltonian
from montecarlo.montecarlo import MonteCarlo
from montecarlo.bitstring import BitString
import networkx as nx
import numpy as np

# 1. Define a graph (e.g., a 1D chain or a 2D grid)
For a 1D chain of 5 sites:
graph = nx.path_graph(5)
For a 2x2 grid:
graph = nx.grid_2d_graph(2, 2)

Example: Set ferromagnetic interaction J_ij = 1 for all edges
for u, v in graph.edges():
    graph.edges[u,v]['weight'] = 1.0

# 2. Initialize the Ising Hamiltonian
ising_system = IsingHamiltonian(G=graph)

Optionally, set external magnetic fields (mu_i)
mus = np.random.uniform(-0.5, 0.5, size=ising_system.N) # Example random fields
ising_system.set_mu(mus)

# 3. Initialize the Monte Carlo simulator
mc_simulator = MonteCarlo(ham=ising_system)

# 4. Run the simulation
T = 2.269  # Example: Critical temperature for 2D Ising model (if J=1)
n_samples = 2000
n_burn = 500
E_samples, M_samples = mc_simulator.run(T=T, n_samples=n_samples, n_burn=n_burn)

# 5. Analyze results
avg_energy = np.mean(E_samples)
avg_magnetization_per_site = np.mean(M_samples) / ising_system.N

print(f"Temperature: {T}")
print(f"Average Energy per site: {avg_energy / ising_system.N}")
print(f"Average Magnetization per site: {avg_magnetization_per_site}")

# For exact calculation (if system size N is small, e.g., N < 15-20)
if ising_system.N <= 10: # Adjust N threshold as needed
    print("\nRunning exact calculation for comparison...")
    E_avg_exact, M_avg_exact, C_exact, chi_exact = ising_system.compute_average_values(T=T)
    print(f"Exact Average Energy per site: {E_avg_exact / ising_system.N}")
    print(f"Exact Average Magnetization per site: {M_avg_exact / ising_system.N}")
    print(f"Exact Heat Capacity per site: {C_exact / ising_system.N}")
    print(f"Exact Magnetic Susceptibility per site: {chi_exact / ising_system.N}")


Running Tests
    To run the tests, navigate to the root directory of the CHEM3684 project and execute:
    pytest
    To include a coverage report:


    pytest --cov=CHEM3684.montecarlo --cov-report=html


Dependencies
    The main dependencies for this package are:
    NumPy
    NetworkX
    These should be automatically installed if you install the package using pip as described above.

Contributing
    Contributions are welcome! Please read the CODE_OF_CONDUCT.md and feel free to submit a pull request or open an issue on the GitHub repository.

License
    This project is licensed under the BSD-3-Clause License - see the LICENSE file for details.

Copyright
    Copyright (c) 2025, Hoon Sung
