import numpy as np
import networkx as nx
from typing import Union
from .bitstring import BitString

class IsingHamiltonian:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.N = len(G.nodes)
        self.mus = np.zeros(self.N)
        self.J = list(G.edges)


    def set_mu(self, mus: np.ndarray):
        """
        Set the chemical potential (external field) for each spin site.
        """
        self.mus = mus
        return self

    def energy(self, config: BitString) -> float:
        """
        Compute the Ising Hamiltonian energy including external magnetic field.
        """
        energy = 0.0
        for (i, j) in self.G.edges:
            Jij = self.G.edges[i, j].get('weight', 1.0)
            si = 1 if config.config[i] == 1 else -1
            sj = 1 if config.config[j] == 1 else -1
            energy += Jij * si * sj

        if self.mus is not None:
            for i in range(len(config.config)):
                si = 1 if config.config[i] == 1 else -1
                energy += self.mus[i] * si

        return energy

    def compute_average_values(self, T: Union[int, float]):
        """
        Compute average energy, magnetization, heat capacity, and susceptibility.
        """
        N = len(self.G.nodes)
        Z = 0.0
        E = M = EE = MM = 0.0

        for i in range(2**N):
            bs = BitString(N)
            bs.set_integer_config(i)
            e = self.energy(bs)
            m = bs.on() - bs.off()
            weight = np.exp(-e / T)
            Z += weight
            E += weight * e
            EE += weight * e**2
            M += weight * m
            MM += weight * m**2

        E /= Z
        EE /= Z
        M /= Z
        MM /= Z

        heat_capacity = (EE - E**2) / (T**2)
        magnetic_susceptibility = (MM - M**2) / T

        return E, M, heat_capacity, magnetic_susceptibility
    
    @property
    def mu(self):
        return self.mus

    @mu.setter
    def mu(self, value):
        self.mus = value

