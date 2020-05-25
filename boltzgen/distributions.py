import torch
import numpy as np
import normflow as nf
import multiprocessing as mp

from . import openmm_interface as omi

from openmmtools.constants import kB


class Boltzmann(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterface.apply
        self.regularize_energy = omi.regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)


class TransformedBoltzmann(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max, transform):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterface.apply
        self.regularize_energy = omi.regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

        self.transform = transform

    def log_prob(self, z):
        z, log_det = self.transform(z)
        return -self.norm_energy(z) + log_det


class TransformedBoltzmannParallel(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution with respect to transformed variables,
    uses OpenMM to get energy and forces and processes the batch of
    states in parallel
    """
    def __init__(self, system, temperature, energy_cut, energy_max, transform,
                 n_threads=None):
        """
        Constructor
        :param system: Molecular system
        :param temperature: Temperature of System
        :param energy_cut: Energy at which logarithm is applied
        :param energy_max: Maximum energy
        :param transform: Coordinate transformation
        :param n_threads: Number of threads to use to process batches, set
        to the number of cpus if None
        """
        # Save input parameters
        self.system = system
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)
        self.n_threads = mp.cpu_count() if n_threads is None else n_threads

        # Create pool for parallel processing
        self.pool = mp.Pool(self.n_threads, omi.OpenMMEnergyInterfaceParallel.var_init,
                            (system, temperature))

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterfaceParallel.apply
        self.regularize_energy = omi.regularize_energy

        self.norm_energy = lambda pos, splits: self.regularize_energy(
            self.openmm_energy(pos, self.pool, splits)[:, 0],
            self.energy_cut, self.energy_max)

        self.transform = transform

    def log_prob(self, z):
        z_, _ = self.transform(z)
        return -self.norm_energy(z_, self.n_threads)


class DoubleWell(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution of the double well potential of the form
    U(x, y) = 1/4 * a * x**4 - 1/2 * b * x**2 + c * x + 1/2 * d * y**2
    """
    def __init__(self, a=1, b=6, c=1, d=1):
        """
        Constructor
        :param a: Parameter of the potential
        :param b: Parameter of the potential
        :param c: Parameter of the potential
        :param d: Parameter of the potential
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def log_prob(self, z):
        return -self.a / 4 * z[:, 0] ** 4 + self.b / 2 * z[:, 0] ** 2 - self.c * z[:, 0] - self.d / 2 * z[:, 1] ** 2