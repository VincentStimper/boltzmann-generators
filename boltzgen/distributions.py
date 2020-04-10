import boltzgen.openmm_interface as omi
import normflow as nf
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
        self.energy_cut = energy_cut
        self.energy_max = energy_max

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterface.apply
        self.regularize_energy = omi.regularize_energy

        self.kbT = (kB * self.temperature)._value
        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature) / self.kbT,
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        return -self.norm_energy(z)
