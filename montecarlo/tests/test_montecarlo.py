import pytest
import numpy as np
import networkx as nx
from montecarlo.montecarlo import MonteCarlo
from montecarlo.ising import IsingHamiltonian
from montecarlo.bitstring import BitString

@pytest.fixture
def simple_ising_hamiltonian():
    """Creates a 2-site ferromagnetic Ising model (J=1, mu=0)."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=1.0) # J_01 = 1.0
    ham = IsingHamiltonian(G)
    # External magnetic field (mu) can be set here if needed
    # ham.set_mu(np.array([0.1, -0.1]))
    return ham

@pytest.fixture
def simple_ising_hamiltonian_af():
    """Creates a 2-site antiferromagnetic Ising model (J=-1, mu=0)."""
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, weight=-1.0) # J_01 = -1.0
    ham = IsingHamiltonian(G)
    return ham

def test_montecarlo_initialization(simple_ising_hamiltonian):
    """Tests if MonteCarlo class initializes correctly with a Hamiltonian object."""
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    assert mc_simulator.ham is simple_ising_hamiltonian
    assert mc_simulator.conf is None

def test_random_bitstring():
    """Tests if random_bitstring method creates a BitString of correct length and content."""
    # A temporary Hamiltonian might be needed to create a MonteCarlo instance,
    # but if random_bitstring doesn't depend on Hamiltonian state, direct call testing is also possible.
    # Assuming here it's called via a MonteCarlo instance.
    G_temp = nx.Graph()
    G_temp.add_node(0) # Minimal graph
    temp_ham = IsingHamiltonian(G_temp)
    mc_simulator = MonteCarlo(ham=temp_ham)

    N = 10
    bs = mc_simulator.random_bitstring(N)
    assert isinstance(bs, BitString), "Returned object must be of BitString type."
    assert len(bs) == N, f"BitString length should be {N}."
    assert all(bit in [0, 1] for bit in bs.config), "All bits in BitString must be 0 or 1."

    N = 5
    bs = mc_simulator.random_bitstring(N)
    assert len(bs) == N

def test_run_output_shape_and_type(simple_ising_hamiltonian):
    """Tests if the run method returns energy and magnetization samples of correct shape and type."""
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    n_samples = 100
    n_burn = 10

    E_samples, M_samples = mc_simulator.run(T=1.0, n_samples=n_samples, n_burn=n_burn)

    assert isinstance(E_samples, np.ndarray), "Energy samples should be a NumPy array."
    assert isinstance(M_samples, np.ndarray), "Magnetization samples should be a NumPy array."
    assert len(E_samples) == n_samples, f"Number of energy samples should be {n_samples}."
    assert len(M_samples) == n_samples, f"Number of magnetization samples should be {n_samples}."

def test_run_burn_in(simple_ising_hamiltonian):
    """Tests if the run method correctly handles the burn-in period (sample collection timing)."""
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    # Set n_samples=0 to check if only burn-in occurs and no samples are collected.
    E_samples, M_samples = mc_simulator.run(T=1.0, n_samples=0, n_burn=50)
    assert len(E_samples) == 0
    assert len(M_samples) == 0

    # Set n_burn=0 to check if sampling occurs at all steps (up to n_samples).
    n_samples = 20
    E_samples, M_samples = mc_simulator.run(T=1.0, n_samples=n_samples, n_burn=0)
    assert len(E_samples) == n_samples
    assert len(M_samples) == n_samples


@pytest.mark.parametrize("temp", [0.01, 1000.0]) # Very low and very high temperatures
def test_run_qualitative_behavior(simple_ising_hamiltonian, temp):
    """
    Tests if the simulation shows qualitatively expected results at very low or high temperatures.
    Checks for trends rather than exact values.
    """
    mc_simulator = MonteCarlo(ham=simple_ising_hamiltonian)
    n_samples = 500 # Sufficient number of samples
    n_burn = 100

    E_samples, M_samples = mc_simulator.run(T=temp, n_samples=n_samples, n_burn=n_burn)

    avg_E = np.mean(E_samples)
    avg_M_abs = np.mean(np.abs(M_samples)) # Average absolute magnetization (for ferromagnet)

    if temp < 0.1: # Low temperature
        # For a 2-site ferromagnet (J=1), ground state energy is -1 (spins: ++ or --)
        # Magnetization |M|=2
        assert avg_E < 0, f"At low temperature (T={temp}), average energy should be negative (current: {avg_E})."
        # Should be close to ground state, so average energy is expected to be near -1
        assert np.isclose(avg_E, -1.0, atol=0.5), f"At low temperature (T={temp}), average energy should be close to -1 (current: {avg_E})."
        assert avg_M_abs > 1.8, f"At low temperature (T={temp}), average absolute magnetization should be close to 2 (current: {avg_M_abs})."
    elif temp > 500: # High temperature
        # Average energy is expected to be near 0 (random spins)
        # Possible energies for J=1: -1 (aligned), +1 (anti-aligned)
        # Average magnetization is expected to be near 0
        assert abs(avg_E) < 0.5, f"At high temperature (T={temp}), average energy should be close to 0 (current: {avg_E})."
        assert avg_M_abs < 0.5, f"At high temperature (T={temp}), average absolute magnetization should be close to 0 (current: {avg_M_abs})."

def test_run_invalid_temperature():
    """Tests if the run method raises an error for invalid temperature (T<=0)."""
    G_temp = nx.Graph()
    G_temp.add_node(0)
    temp_ham = IsingHamiltonian(G_temp)
    mc_simulator = MonteCarlo(ham=temp_ham)
    with pytest.raises(ValueError, match="Temperature T must be positive."):
        mc_simulator.run(T=0)
    with pytest.raises(ValueError, match="Temperature T must be positive."):
        mc_simulator.run(T=-1.0)

def test_run_invalid_samples_burn():
    """Tests if the run method raises an error for invalid n_samples or n_burn values."""
    G_temp = nx.Graph()
    G_temp.add_node(0)
    temp_ham = IsingHamiltonian(G_temp)
    mc_simulator = MonteCarlo(ham=temp_ham)

    # If n_samples=0 is valid (as in test_run_burn_in), this part needs adjustment.
    with pytest.raises(ValueError, match="Number of samples"):
        mc_simulator.run(n_samples=0, n_burn=10)

    with pytest.raises(ValueError, match="Number of samples"):
        mc_simulator.run(n_samples=-10)

    with pytest.raises(ValueError, match="Number of burn-in steps"):
        mc_simulator.run(n_burn=-10)
