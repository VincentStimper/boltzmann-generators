import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '../../normalizing-flows')
import normflows as nf

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from simtk.openmm.app import StateDataReporter
import mdtraj

sys.path.insert(0, '../')
import boltzgen as bg


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processID', type=int, default=0)
    args = parser.parse_args()

    # log mass 0 stepsize 0.05
    ndim = 66
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [6, 8, 9, 10, 14]
    temperature = 1000

    system = testsystems.AlanineDipeptideVacuum(constraints=None)
    sim = app.Simulation(system.topology, system.system,
        mm.LangevinIntegrator(temperature * unit.kelvin,
                            1. / unit.picosecond,
                            1. * unit.femtosecond),
        mm.Platform.getPlatformByName('CPU'))


    # Load the training data
    training_data_traj = mdtraj.load('saved_data/aldp_vacuum_without_const.h5')
    training_data_traj.center_coordinates()
    ind = training_data_traj.top.select("backbone")
    training_data_traj.superpose(training_data_traj, 0, atom_indices=ind,
        ref_atom_indices=ind)
    # Gather the training data into a pytorch Tensor with the right shape
    training_data = training_data_traj.xyz
    n_atoms = training_data.shape[1]
    n_dim = n_atoms * 3
    training_data_npy = training_data.reshape(-1, n_dim)
    training_data = torch.from_numpy(
        training_data_npy.astype("float64"))

    coord_transform = bg.flows.CoordinateTransform(training_data,
        n_dim, z_matrix, cart_indices)
    target_dist = bg.distributions.TransformedBoltzmannParallel(
        system, temperature,
        energy_cut=1.e+2,
        energy_max=1.e+20,
        transform=coord_transform,
        n_threads=3)
        

    prior = nf.distributions.DiagGaussian(60, trainable=False)
    prior.double()
    beta = np.linspace(1.0, 0.0, num=50)
    hais = nf.HAIS.HAIS(beta, prior, target_dist, num_leapfrog=5,
        step_size=0.05 * torch.ones(60, dtype=torch.float64),
        log_mass = torch.zeros(60, dtype=torch.float64))

    print("sampling")
    for i in range(3):
        print(i)
        samples, logweights = hais.sample(10)
        samples = samples.detach().numpy()
        logweights = logweights.detach().numpy()
        np.save('saved_data/temp/samples_{}'.format(i) + '_process_' + \
            str(args.processID), samples)
        np.save('saved_data/temp/logweights_{}'.format(i) + '_process_' + \
            str(args.processID), logweights)


if __name__ == '__main__':
    main()


# %%
