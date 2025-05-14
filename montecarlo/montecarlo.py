import numpy as np
import random
import copy as cp
from .bitstring import BitString
from .ising import IsingHamiltonian

class MonteCarlo:
    """
    Class to perform Metropolis Monte Carlo simulation on an Ising Hamiltonian.

    Attributes
    ----------
    ham : IsingHamiltonian
        The Ising Hamiltonian object defining the system.
    conf : BitString or None
        The current spin configuration of the system. Initialized to None,
        and set during the `run` method or by other methods.
    """

    def __init__(self, ham: IsingHamiltonian):
        """
        Initializes the MonteCarlo simulator with a given Hamiltonian.

        Parameters
        ----------
        ham : IsingHamiltonian
            An instance of the IsingHamiltonian class representing the system
            to be simulated.
        """
        self.ham = ham
        self.conf = None # Current configuration, initialized during run

    def run(self, T: float = 1.0, n_samples: int = 1000, n_burn: int = 100):
        """
        Run the Metropolis Monte Carlo simulation.

        Starts from a random configuration, performs burn-in steps,
        and then collects samples for energy and magnetization.

        Parameters
        ----------
        T : float, optional
            Temperature for the simulation (default is 1.0). Must be > 0.
        n_samples : int, optional
            Number of samples to collect after burn-in (default is 1000).
        n_burn : int, optional
            Number of burn-in steps to equilibrate the system (default is 100).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing two NumPy arrays:
            - E_samples (np.ndarray): Collected energy samples.
            - M_samples (np.ndarray): Collected magnetization samples.
        """
        if T <= 0:
            raise ValueError("Temperature T must be positive.")
        if n_samples <= 0:
            raise ValueError("Number of samples (n_samples) must be positive.")
        if n_burn < 0:
            raise ValueError("Number of burn-in steps (n_burn) cannot be negative.")

        N = self.ham.N # Number of sites from the Hamiltonian
        self.conf = self.random_bitstring(N) # Initialize with a random configuration

        E_samples = []
        M_samples = []

        current_energy = self.ham.energy(self.conf) # Initial energy

        for step in range(n_samples + n_burn):
            # Attempt N spin flips (one MC sweep)
            for _ in range(N): # The variable 'i' for site index was used to flip, but now a random site is chosen
                site_to_flip = random.randrange(N) # Choose a random site to flip

                # Energy before the proposed flip (already known)
                # old_site_energy_contribution = ... (for local update, if implemented)

                # Propose flip
                self.conf.flip_site(site_to_flip)
                new_energy = self.ham.energy(self.conf) # Energy after the proposed flip

                delta_E = new_energy - current_energy

                # Metropolis acceptance rule
                if delta_E <= 0 or random.uniform(0, 1) < np.exp(-delta_E / T):
                    current_energy = new_energy  # Accept the new configuration
                else:
                    self.conf.flip_site(site_to_flip)  # Reject: flip back to the old configuration
                                                      # No need to re-calculate energy, it's current_energy

            # Collect samples after burn-in period
            if step >= n_burn:
                # Energy is already `current_energy`
                # Magnetization: sum of spins (+1 or -1 for each site)
                m = sum([1 if bit == 1 else -1 for bit in self.conf.config])
                E_samples.append(current_energy)
                M_samples.append(m)

        return np.array(E_samples), np.array(M_samples)

    def random_bitstring(self, N: int) -> BitString:
        """
        Generates a BitString of length N with a random configuration.

        Each bit is independently chosen to be 0 or 1 with equal probability.

        Parameters
        ----------
        N : int
            The desired length of the BitString.

        Returns
        -------
        BitString
            A BitString object with a random configuration.
        """
        if N <= 0:
            raise ValueError("Length N must be positive.")
        bs = BitString(N)
        # BitString.__init__ might already initialize randomly.
        # If not, or if a specific random initialization is needed:
        bs.config = np.random.randint(0, 2, size=N)
        # The loop below is an alternative if BitString.config needs to be set bit by bit
        # for i in range(N):
        #     bs.config[i] = random.choice([0, 1])
        return bs
