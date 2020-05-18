import normflow as nf

class BoltzmannGenerator(nf.NormalizingFlow):
    """
    Boltzmann Generator with architecture inspired by arXiv:2002.06707
    """
    def __init__(self, system='AlanineDipeptideImplicit', temperature=1000, energy_cut=1e2,
                 energy_max=1e20, data_transform=None, force_platform='CUDA', force_precision='double',
                 rnvp_blocks=5, actnorm=True, mcmc_layer=False, mcmc_steps=20, proposal_std=0.1,
                 latent_size=60, hidden_layers=3, hidden_units=128):
        """
        Constructor
        :param system: String, specifying the system from which states shall be generated
        :param temperature: Float or Double, temperature of the system
        :param energy_cut: Float or Double, energy level at which regularization shall be applied
        :param energy_max: Float or Double, maximum level at which energies will be clamped
        :param data_transform: torch float or double tensor, data to use to initialize transform
        :param force_platform: String, specifies the platform used for force and energy calculations
        :param force_precision: String, specifies the precision used for force and energy calculations
        :param rnvp_blocks: Int, number of Real NVP blocks, consisting of two alternating Real NVP
        layers each, to use
        :param actnorm:
        :param mcmc_layer:
        :param mcmc_steps:
        :param proposal_std:
        :param latent_size:
        :param hidden_layers:
        :param hidden_units:
        """