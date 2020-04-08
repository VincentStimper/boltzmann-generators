import boltzgen.openmm_interface as omi
import normflow as nf
from openmmtools.constants import kB


class Boltzmann(nf.distributions.PriorDistribution):
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        self.sim_context = sim_context
        self.temperature = temperature
        self.openmm_energy = omi.OpenMMEnergyInterface.apply
        self.kbT = (kB * self.temperature)._value
        self.norm_energy = lambda pos: self.openmm_energy(pos, self.sim_context, temperature) / self.kbT

    def log_prob(self, z):
        return -self.norm_energy(z)
