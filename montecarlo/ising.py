import numpy as np
import networkx as nx
from typing import Union
from .bitstring import BitString

class IsingHamiltonian:
    """
    Represents an Ising Hamiltonian for a given graph.

    H = - sum J_ij s_i s_j - sum mu_i s_i

    Attributes
    ----------
    G : nx.Graph
        The graph representing the lattice structure. Edge weights are J_ij.
    N : int
        Number of sites (nodes) in the graph.
    mus : np.ndarray
        Array of chemical potentials (external fields) for each site.
    J : list
        List of edges from the graph, representing interactions.
        Note: Interaction strengths J_ij are taken from G.edges[i,j].get('weight', 1.0).
    """
    def __init__(self, G: nx.Graph):
        """
        Initializes the IsingHamiltonian with a graph.

        Parameters
        ----------
        G : nx.Graph
            The graph defining the lattice and interactions.
            Interaction strengths J_ij can be set as 'weight' attributes on edges.
            If no 'weight' is specified, J_ij defaults to 1.0.
        """
        self.G = G
        self.N = len(G.nodes)
        self.mus = np.zeros(self.N) # External fields, mu_i
        # J_ij are implicitly stored as edge weights in self.G
        # self.J here is just a list of edge tuples, not the interaction strengths themselves.
        # This might be slightly confusing; consider renaming or clarifying its role if used elsewhere.
        self.J = list(G.edges)


    def set_mu(self, mus: np.ndarray):
        """
        Set the chemical potential (external field) for each spin site.

        Parameters
        ----------
        mus : np.ndarray
            An array of chemical potentials for each site.
            Its length should match the number of sites N.

        Returns
        -------
        IsingHamiltonian
            The instance itself, allowing for method chaining.
        """
        if len(mus) != self.N:
            raise ValueError(f"Length of mus ({len(mus)}) must match number of sites N ({self.N}).")
        self.mus = mus
        return self

    def energy(self, config: BitString) -> float:
        """
        Compute the Ising Hamiltonian energy for a given spin configuration.

        Energy = - sum_edges J_ij * s_i * s_j - sum_sites mu_i * s_i
        where s_i = +1 if bit is 1, and s_i = -1 if bit is 0.

        Parameters
        ----------
        config : BitString
            A BitString object representing the spin configuration.
            Bits of 1 are treated as spin +1, bits of 0 as spin -1.

        Returns
        -------
        float
            The total energy of the configuration.
        """
        energy_val = 0.0 # Renamed to avoid conflict with method name

        # Interaction term
        for (i, j) in self.G.edges:
            Jij = self.G.edges[i, j].get('weight', 1.0) # Default J_ij = 1.0
            si = 1 if config.config[i] == 1 else -1
            sj = 1 if config.config[j] == 1 else -1
            energy_val -= Jij * si * sj # Ising convention usually has a minus sign here

        # External field term
        # self.mus is initialized as np.zeros(self.N), so it's not None unless explicitly set to None.
        # The original check `if self.mus is not None:` might be redundant if self.mus always exists.
        # However, keeping it if there's a scenario where self.mus could be None.
        if self.mus is not None and len(self.mus) == self.N:
            for i in range(self.N): # Iterate up to N, not len(config.config) for safety
                si = 1 if config.config[i] == 1 else -1
                energy_val -= self.mus[i] * si # Ising convention usually has a minus sign here
        elif self.mus is not None and len(self.mus) != self.N:
            # This case should ideally be prevented by checks in set_mu or __init__
            print(f"Warning: Length of mus ({len(self.mus)}) does not match N ({self.N}). External field term skipped or partially applied.")


        return energy_val

    def compute_average_values(self, T: Union[int, float]):
        """
        Compute canonical ensemble average values by exact enumeration.

        Calculates average energy, magnetization, heat capacity,
        and magnetic susceptibility at a given temperature T.

        Note: This method performs an exact enumeration over all 2**N states.
        It will be very slow for N > ~20.

        Parameters
        ----------
        T : float or int
            The temperature for the ensemble average. Must be > 0.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing:
            - E_avg (float): Average energy.
            - M_avg (float): Average magnetization.
            - C (float): Heat capacity.
            - chi (float): Magnetic susceptibility.
        """
        if T <= 0:
            raise ValueError("Temperature T must be positive.")

        # N_sites = self.N # Use self.N consistently
        Z = 0.0  # Partition function
        E_sum = 0.0  # Sum of E * exp(-E/T)
        M_sum = 0.0  # Sum of M * exp(-E/T)
        EE_sum = 0.0 # Sum of E^2 * exp(-E/T)
        MM_sum = 0.0 # Sum of M^2 * exp(-E/T)

        # Iterate over all 2**N possible spin configurations
        for i in range(2**self.N):
            bs = BitString(self.N)
            bs.set_integer_config(i) # Set configuration based on integer i
            
            e_current = self.energy(bs)
            # Magnetization: sum of spins (+1 or -1)
            m_current = sum([1 if bit == 1 else -1 for bit in bs.config])
            
            weight = np.exp(-e_current / T)
            
            Z += weight
            E_sum += weight * e_current
            EE_sum += weight * e_current**2
            M_sum += weight * m_current
            MM_sum += weight * m_current**2

        if Z == 0: # Avoid division by zero if all weights are zero (e.g., T is extremely low and energies are high)
            return 0.0, 0.0, 0.0, 0.0

        E_avg = E_sum / Z
        EE_avg = EE_sum / Z
        M_avg = M_sum / Z
        MM_avg = MM_sum / Z

        # Heat Capacity C = (<E^2> - <E>^2) / T^2
        heat_capacity = (EE_avg - E_avg**2) / (T**2)
        # Magnetic Susceptibility chi = (<M^2> - <M>^2) / T
        magnetic_susceptibility = (MM_avg - M_avg**2) / T

        return E_avg, M_avg, heat_capacity, magnetic_susceptibility

    @property
    def mu(self):
        """numpy.ndarray: External magnetic fields (chemical potentials) for each site."""
        return self.mus

    @mu.setter
    def mu(self, value: np.ndarray):
        """
        Sets the external magnetic fields (chemical potentials).

        Parameters
        ----------
        value : np.ndarray
            An array of chemical potentials. Must have length N.
        """
        if len(value) != self.N:
            raise ValueError(f"Length of mu values ({len(value)}) must match number of sites N ({self.N}).")
        self.mus = value