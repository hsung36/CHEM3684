import pytest
import numpy as np
import networkx as nx
from montecarlo.montecarlo import MonteCarlo
from montecarlo.ising import IsingHamiltonian
from montecarlo.bitstring import BitString

@pytest.fixture
def simple_ising_hamiltonian():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=1.0)
    ham = IsingHamiltonian(G)
    return ham

@pytest.fixture
def simple_ising_hamiltonian_af():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=-1.0)
    ham = IsingHamiltonian(G)
    return ham

def test_montecarlo_initialization(simple_ising_hamiltonian):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    assert mc_simulator.ham is simple_ising_hamiltonian
    assert mc_simulator.conf is None

def test_random_bitstring():
    G_temp = nx.Graph()
    G_temp.add_node(0)
    temp_ham = IsingHamiltonian(G_temp)
    mc_simulator = MonteCarlo(ham=temp_ham)

    N = 10
    bs = mc_simulator.random_bitstring(N)
    assert isinstance(bs, BitString)
    assert len(bs) == N
    assert all(bit in [0, 1] for bit in bs.config)

    N = 5
    bs = mc_simulator.random_bitstring(N)
    assert len(bs) == N

def test_run_output_shape_and_type(simple_ising_hamiltonian):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    n_samples = 100
    n_burn = 10

    E_samples, M_samples = mc_simulator.run(T=1.0, n_samples=n_samples, n_burn=n_burn)

    assert isinstance(E_samples, np.ndarray)
    assert isinstance(M_samples, np.ndarray)
    assert len(E_samples) == n_samples
    assert len(M_samples) == n_samples

def test_run_burn_in_effect(simple_ising_hamiltonian):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    n_samples_for_test = 20
    E_samples_no_burn, M_samples_no_burn = mc_simulator.run(T=1.0, n_samples=n_samples_for_test, n_burn=0)
    assert len(E_samples_no_burn) == n_samples_for_test
    assert len(M_samples_no_burn) == n_samples_for_test

    n_burn_steps = 30
    E_samples_with_burn, M_samples_with_burn = mc_simulator.run(T=1.0, n_samples=n_samples_for_test, n_burn=n_burn_steps)
    assert len(E_samples_with_burn) == n_samples_for_test
    assert len(M_samples_with_burn) == n_samples_for_test


@pytest.mark.parametrize("temp", [0.01, 1000.0])
def test_run_qualitative_behavior(simple_ising_hamiltonian, temp):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    n_samples = 2000
    n_burn = 500

    E_samples, M_samples = mc_simulator.run(T=temp, n_samples=n_samples, n_burn=n_burn)

    avg_E = np.mean(E_samples)
    avg_M_abs = np.mean(np.abs(M_samples))

    if temp < 0.1:
        assert avg_E < 0, f"At low T={temp}, avg_E should be < 0 (was {avg_E})."
        assert np.isclose(avg_E, -1.0, atol=0.3), f"At low T={temp}, avg_E should be close to -1.0 (was {avg_E})."
        assert avg_M_abs > 1.7, f"At low T={temp}, avg_M_abs should be close to 2 (was {avg_M_abs})."
    elif temp > 500:
        assert np.isclose(avg_E, 0.0, atol=0.75), f"At high T={temp}, avg_E should be close to 0.0 (was {avg_E})."
        assert np.isclose(avg_M_abs, 1.0, atol=0.5), f"At high T={temp}, avg_M_abs should be close to 1.0 (was {avg_M_abs})."

def test_run_invalid_temperature(simple_ising_hamiltonian):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    with pytest.raises(ValueError, match="Temperature T must be positive."):
        mc_simulator.run(T=0)
    with pytest.raises(ValueError, match="Temperature T must be positive."):
        mc_simulator.run(T=-1.0)

def test_run_invalid_samples_burn(simple_ising_hamiltonian):
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)

    with pytest.raises(ValueError, match="Number of samples \\(n_samples\\) must be positive."):
        mc_simulator.run(n_samples=0)

    with pytest.raises(ValueError, match="Number of samples \\(n_samples\\) must be positive."):
        mc_simulator.run(n_samples=-10)

    with pytest.raises(ValueError, match="Number of burn-in steps \\(n_burn\\) cannot be negative."):
        mc_simulator.run(n_burn=-10)
