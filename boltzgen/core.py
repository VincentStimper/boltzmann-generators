import torch
import numpy as np

import normflow as nf

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
import mdtraj

from boltzgen.flows import CoordinateTransform
from boltzgen.distributions import Boltzmann, TransformedBoltzmann

class BoltzmannGenerator(nf.NormalizingFlow):
    """
    Boltzmann Generator with architecture inspired by arXiv:2002.06707
    """
    def __init__(self, config):
        """
        Constructor
        :param config: Dict, specified by a yaml file, see sample config file
        """

        self.config = config
        # Set up simulation object
        if config['system']['name'] == 'AlanineDipeptideVacuum':
            ndim = 66
            z_matrix = [
                (1, [4, 5, 6]),
                (0, [1, 4, 5]),
                (2, [1, 0, 4]),
                (3, [1, 0, 2]),
                (7, [6, 4, 5]),
                (9, [8, 6, 7]),
                (10, [8, 6, 9]),
                (11, [10, 8, 9]),
                (12, [10, 8, 11]),
                (13, [10, 11, 12]),
                (17, [16, 14, 15]),
                (19, [18, 16, 17]),
                (20, [18, 19, 16]),
                (21, [18, 19, 20])
            ]
            backbone_indices = [4, 5, 6, 8, 14, 15, 16, 18]
            temperature = config['system']['temperature']

            self.system = testsystems.AlanineDipeptideVacuum()
            if config['system']['platform'] == 'CPU':
                self.sim = app.Simulation(self.system.topology, self.system.system,
                                          mm.LangevinIntegrator(temperature * unit.kelvin,
                                                                1. / unit.picosecond,
                                                                1. * unit.femtosecond),
                                          mm.Platform.getPlatformByName('CPU'))
            else:
                self.sim = app.Simulation(self.system.topology, self.system.system,
                                          mm.LangevinIntegrator(temperature * unit.kelvin,
                                                                1. / unit.picosecond,
                                                                1. * unit.femtosecond),
                                          mm.Platform.getPlatformByName(config['system']['platform']),
                                          {'Precision': config['system']['precision']})

            training_data = torch.randn(66, 66)
        else:
            raise NotImplementedError('The system ' + config['system']['name']
                                      + ' has not been implemented.')

        # Load data for transform if specified
        if config['data_path'] is not None:
            # Load the alanine dipeptide trajectory
            traj = mdtraj.load(config['data_path'])
            traj.center_coordinates()

            # superpose on the backbone
            ind = traj.top.select("backbone")

            traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

            # Gather the training data into a pytorch Tensor with the right shape
            training_data = traj.xyz
            n_atoms = training_data.shape[1]
            n_dim = n_atoms * 3
            training_data_npy = training_data.reshape(-1, n_dim)
            training_data = torch.from_numpy(training_data_npy.astype("float64"))

        # Set up model
        # Define flows
        rnvp_blocks = config['model']['rnvp']['blocks']

        # Set prior and q0
        energy_cut = config['system']['energy_cut']
        energy_max = config['system']['energy_max']
        p = Boltzmann(self.sim.context, temperature, energy_cut=energy_cut,
                      energy_max=energy_max)
        transform = CoordinateTransform(training_data, ndim, z_matrix, backbone_indices)
        if config['model']['snf']['mcmc']:
            p_ = TransformedBoltzmann(self.sim.context, temperature, energy_cut=energy_cut,
                                      energy_max=energy_max, transform=transform)

        latent_size = config['model']['latent_size']
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)

        # Set up flow layers
        hidden_units = config['model']['rnvp']['hidden_units']
        hidden_layers = config['model']['rnvp']['hidden_layers']
        output_fn = config['model']['rnvp']['output_fn']
        output_scale = config['model']['rnvp']['output_scale']
        init_zeros = config['model']['rnvp']['init_zeros']

        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
        flows = []
        for i in range(rnvp_blocks):
            # Two alternating Real NVP layers
            s = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size],
                            output_fn=output_fn, output_scale=output_scale, init_zeros=init_zeros)
            t = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size])
            flows += [nf.flows.MaskedAffineFlow(b, s, t)]
            s = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size],
                            output_fn=output_fn, output_scale=output_scale, init_zeros=init_zeros)
            t = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size])
            flows += [nf.flows.MaskedAffineFlow(1 - b, s, t)]

            # ActNorm
            if config['model']['actnorm']:
                flows += [nf.flows.ActNorm(latent_size)]

            # MCMC layer
            if config['model']['snf']['mcmc']:
                if 'lambda' in config['model']['snf'].keys():
                    lam = config['model']['snf']['lambda'][i]
                else:
                    lam = (i + 1) / rnvp_blocks
                dist = nf.distributions.LinearInterpolation(p_, q0, lam)
                prop_scale = config['model']['snf']['proposal_std'] * np.ones(latent_size)
                proposal = nf.distributions.DiagGaussianProposal((latent_size,), prop_scale)
                flows += [nf.flows.MetropolisHastings(dist, proposal,
                                                      config['model']['snf']['steps'])]
        # Coordinate transformation
        flows += [transform]

        # Construct flow model
        super().__init__(q0=q0, flows=flows, p=p)

