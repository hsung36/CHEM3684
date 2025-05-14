import numpy as np
import random
import copy as cp
from .bitstring import BitString

class MonteCarlo:
    """
    Class to perform Metropolis Monte Carlo simulation on an Ising Hamiltonian.
    """

    def __init__(self, ham):
        self.ham = ham
        self.conf = None
        
    def run(self, T=1, n_samples=1000, n_burn=100):
        N = len(self.ham.G.nodes)
        self.conf = self.random_bitstring(N)
        
        E_samples = []
        M_samples = []
        
        for step in range(n_samples + n_burn):
            for i in range(N):
                old_energy = self.ham.energy(self.conf)
                
                # Propose flip
                self.conf.flip_site(i)
                new_energy = self.ham.energy(self.conf)
                
                delta_E = new_energy - old_energy

                # Metropolis acceptance
                if delta_E <= 0 or random.uniform(0, 1) < np.exp(-delta_E / T):
                    old_energy = new_energy  # Accept
                else:
                    self.conf.flip_site(i)  # Reject

            if step >= n_burn:
                e = self.ham.energy(self.conf)
                m = self.conf.on() - self.conf.off()
                E_samples.append(e)
                M_samples.append(m)
        
        return np.array(E_samples), np.array(M_samples)

    def random_bitstring(self, N):
        bs = BitString(N)
        for i in range(N):
            bs.config[i] = random.choice([0, 1])
        return bs
