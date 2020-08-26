# Takes a config file specifying the models and experiments and performs the
# training/experiments as specified


#%%
import torch
import torch.nn as nn
import numpy as np
import math
import sys
sys.path.insert(0, '../../normalizing-flows')
import normflow as nf

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from simtk.openmm.app import StateDataReporter
import mdtraj

sys.path.insert(0, '../')
import boltzgen as bg

import matplotlib.pyplot as plt

import os.path
import os
from sys import stdout

import argparse

from tqdm import tqdm

from time import time

def main():

    parser = argparse.ArgumentParser(description="Run experiments/generate samples from HMC chains")

    parser.add_argument('--config', type=str,
                        help='Path to config file specifying the experiment details',
                        default='../config/HMC.yaml')
    parser.add_argument('--processID', type=int,
                        help='When generating batches of samples in parallel, this ID can be appended to file names to differentiate between processes',
                        default=0)

    args = parser.parse_args()

    config = bg.utils.get_config(args.config)

    class FlowHMC(nn.Module):
        """
        Model with a normalizing flow as the initial distribution of a HMC chain
        """
        def __init__(self, config):
            super().__init__()
            self.config = config
            # Setup the simulation object
            if config['system']['name'] == 'AlanineDipeptideVacuum':
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
                temperature = config['system']['temperature']

                system = testsystems.AlanineDipeptideVacuum(constraints=None)
                if config['system']['platform'] == 'CPU':
                    sim = app.Simulation(system.topology, system.system,
                        mm.LangevinIntegrator(temperature * unit.kelvin,
                                            1. / unit.picosecond,
                                            1. * unit.femtosecond),
                        mm.Platform.getPlatformByName('CPU'))
                elif config['system']['platform'] == 'Reference':
                    sim = app.Simulation(system.topology, system.system,
                        mm.LangevinIntegrator(temperature * unit.kelvin,
                                            1. / unit.picosecond,
                                            1. * unit.femtosecond),
                        mm.Platform.getPlatformByName('Reference'))
                else:
                    sim = app.Simulation(system.topology, system.system,
                        mm.LangevinIntegrator(temperature * unit.kelvin,
                                            1. / unit.picosecond,
                                            1. * unit.femtosecond),
                        mm.Platform.getPlatformByName(config['system']['platform']),
                        {'Precision': config['system']['precision']})
            else:
                raise NotImplementedError('The system ' + config['system']['name']
                                            + ' has not been implemented.')


            # Simulate the training data if not done already
            if not os.path.exists(config['system']['training_data_path']):
                print("generating training data as file ",
                    config['system']['training_data_path'], " does not exist")
                sim.context.setPositions(system.positions)
                sim.minimizeEnergy()
                sim.reporters.append(mdtraj.reporters.HDF5Reporter(
                    config['system']['training_data_path'], 10))
                sim.reporters.append(StateDataReporter(stdout, 100000, step=True,
                        potentialEnergy=True, temperature=True))
                sim.step(1000000)
                sim.reporters[0].close()

            # Load the training data
            self.training_data_traj = mdtraj.load(config['system']['training_data_path'])
            self.training_data_traj.center_coordinates()
            ind = self.training_data_traj.top.select("backbone")
            self.training_data_traj.superpose(self.training_data_traj, 0, atom_indices=ind,
                ref_atom_indices=ind)
            # Gather the training data into a pytorch Tensor with the right shape
            training_data = self.training_data_traj.xyz
            n_atoms = training_data.shape[1]
            n_dim = n_atoms * 3
            training_data_npy = training_data.reshape(-1, n_dim)
            self.training_data = torch.from_numpy(
                training_data_npy.astype("float64"))

            coord_transform = bg.flows.CoordinateTransform(self.training_data,
                n_dim, z_matrix, cart_indices)
            if config['system']['parallel_energy']:
                self.target_dist = bg.distributions.TransformedBoltzmannParallel(
                    system, temperature,
                    energy_cut=config['system']['energy_cut'],
                    energy_max=config['system']['energy_max'],
                    transform=coord_transform,
                    n_threads=config['system']['n_threads'])
            else:
                self.target_dist = bg.distributions.TransformedBoltzmann(
                    sim.context, temperature,
                    energy_cut=config['system']['energy_cut'],
                    energy_max=config['system']['energy_max'],
                    transform=coord_transform)



            # Set up flow layers
            latent_size = config['initial_dist_model']['latent_size']
            rnvp_blocks = config['initial_dist_model']['rnvp']['blocks']
            hidden_units = config['initial_dist_model']['rnvp']['hidden_units']
            hidden_layers = config['initial_dist_model']['rnvp']['hidden_layers']
            output_fn = config['initial_dist_model']['rnvp']['output_fn']
            output_scale = config['initial_dist_model']['rnvp']['output_scale']
            init_zeros = config['initial_dist_model']['rnvp']['init_zeros']

            self.rnvp_initial_dist = nf.distributions.DiagGaussian(
                latent_size, trainable=False)

            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            raw_flows = []
            for i in range(rnvp_blocks):
                # Two alternating Real NVP layers
                s = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size],
                        output_fn=output_fn, output_scale=output_scale, init_zeros=init_zeros)
                t = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size])
                raw_flows += [nf.flows.MaskedAffineFlow(b, s, t)]
                s = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size],
                        output_fn=output_fn, output_scale=output_scale, init_zeros=init_zeros)
                t = nf.nets.MLP([latent_size] + hidden_layers * [hidden_units] + [latent_size])
                raw_flows += [nf.flows.MaskedAffineFlow(1 - b, s, t)]

                # ActNorm
                if config['initial_dist_model']['actnorm']:
                    raw_flows += [nf.flows.ActNorm(latent_size)]


            if config['initial_dist_model']['scaling'] is not None:
                print("adding scaling layer, using scaling:",
                    config['initial_dist_model']['scaling'])
                with torch.no_grad():
                    x, _ = self.rnvp_initial_dist.forward(100000)
                    for i in range(len(raw_flows)):
                        x, _ = raw_flows[i].forward(x)
                    raw_flows += [bg.flows.Scaling(torch.mean(x, 0),
                        torch.log(torch.tensor(config['initial_dist_model']['scaling'])))]

            if config['initial_dist_model']['noise_std'] is not None:
                print("adding Gaussian noise layer, using noise std:",
                    config['initial_dist_model']['noise_std'])
                raw_flows += [bg.flows.AddNoise(
                    torch.tensor(math.log(config['initial_dist_model']['noise_std'])))]

            self.end_init_flow_idx = len(raw_flows)

            # Add HMC layers
            for i in range(config['hmc']['chain_length']):
                step_size = config['hmc']['starting_step_size'] * \
                    torch.ones((latent_size,))
                log_step_size = torch.log(step_size)
                log_mass = config['hmc']['starting_log_mass'] * \
                    torch.ones((latent_size,))
                raw_flows += [nf.flows.HamiltonianMonteCarlo(self.target_dist,
                    config['hmc']['leapfrog_steps'], log_step_size, log_mass)]

            raw_flows += [coord_transform]

            self.flows = nn.ModuleList(raw_flows)


        def load_model_for_initial_flow(self, config, reporting=False):
            # Loads parameters into the inital flow from the given path
            print("Loading initial flow model:",
                config['initial_dist_model']['load_model_path'])
            loaded_param_dict = torch.load(config['initial_dist_model']['load_model_path'],
                map_location=torch.device(config['initial_dist_model']['device']))
            with torch.no_grad():
                state_dict = self.state_dict()
                for param in self.named_parameters():
                    try:
                        state_dict[param[0]] = loaded_param_dict[param[0]]
                    except:
                        if reporting:
                            print(param[0], "of flowhmc model not loaded")
                for buffer in self.named_buffers():
                    # We want to load the coord transform parameters so will 
                    # need to adjust the flow number
                    self_buffer_name = buffer[0]
                    if "mixed_transform" in self_buffer_name:
                        load_buffer_name = "flows." + \
                            str(config['initial_dist_model']\
                                ['final_flow_idx_for_loaded_model']) + \
                            self_buffer_name[8:]
                        if reporting:
                            print("changed name from ", self_buffer_name, "to",
                                load_buffer_name)
                    else:
                        load_buffer_name = self_buffer_name
                    try:
                        state_dict[self_buffer_name] = \
                            loaded_param_dict[load_buffer_name]
                    except:
                        if reporting:
                            print(self_buffer_name, "of flowhmc model not loaded")

                self.load_state_dict(state_dict)

        def sample_initial_dist(self, num_samples):
            # Draw samples from the initial distribution to the HMC chain
            x, _ = self.rnvp_initial_dist.forward(num_samples)
            for i in range(self.end_init_flow_idx):
                x, _ = self.flows[i].forward(x)
            return x
        
        def sample(self, num_samples, require_grad=True):
            # Draw samples from the full initial distribution plus hmc flow
            x, _ = self.rnvp_initial_dist.forward(num_samples)
            for flow in self.flows:
                if not require_grad:
                    x = x.detach()
                x, _ = flow.forward(x)
            return x

        def get_hmc_parameters(self):
            hmc_parameters = []
            for i in range(self.end_init_flow_idx, len(self.flows)-1, 1):
                for p in self.flows[i].parameters():
                    hmc_parameters.append(p)
            return hmc_parameters

        def eval_log_target(self, num_samples):
            # Evaulates the log target at the end of the chain averaging over the
            # given number of samples
            x, _ = self.rnvp_initial_dist.forward(num_samples)

            # Sample up to just before the coord transform
            for i in range(len(self.flows)-1):
                x, _ = self.flows[i].forward(x)
            
            log_probs = self.target_dist.log_prob(x)
            return torch.mean(log_probs)



    flowhmc = FlowHMC(config)
    flowhmc = flowhmc.double()
    flowhmc.load_model_for_initial_flow(config)

    # Load a full model
    if config['full_model_path'] is not None:
        print("Loading full model: ", config['full_model_path'])
        flowhmc.load_state_dict(torch.load(config['full_model_path']))

    # Train with Ergodic Inference
    if config['ei_training']['do_training']:
        print("Training with EI")
        optimizer = torch.optim.Adam(flowhmc.get_hmc_parameters(),
            lr=config['ei_training']['lr'])
        losses = np.array([])
        save_name_base = "hmc_ei"

        if config['ei_training']['resume']:
            latest_cp = bg.utils.get_latest_checkpoint(
                config['ei_training']['save_path'], 'model')
            if latest_cp is not None:
                start_iter = int(latest_cp[-8:-3])
                flowhmc.load_state_dict(torch.load(latest_cp))
                optimizer_path = config['ei_training']['save_path'] + \
                    "optimizer_" + save_name_base + ".pt"
                if os.path.exists(optimizer_path):
                    optimizer.load_state_dict(torch.load(optimizer_path))
                trainprog_path = config['ei_training']['save_path'] + 'trainprog_' + \
                    save_name_base + "_ckpt_%05i.txt" % start_iter
                if os.path.exists(trainprog_path):
                    losses = np.loadtxt(trainprog_path)
            else:
                start_iter = 0
        else:
            start_iter = 0

        start_time = time()
        
        print("Iter    Loss")
        for iter in range(start_iter, config['ei_training']['iters']):
            optimizer.zero_grad()
            loss = -flowhmc.eval_log_target(config['ei_training']['samples_per_iter'])
            if not torch.isnan(loss):
                loss.backward()
            optimizer.step()

            losses = np.append(losses, loss.detach().numpy())
            print(iter, loss.detach().numpy())

            if (iter+1)%config['ei_training']['save_interval'] == 0:
                np.savetxt(config['ei_training']['save_path'] + "trainprog_" + \
                    save_name_base + "_ckpt_%05i.txt"%(iter+1), losses)
                torch.save(flowhmc.state_dict(), config['ei_training']['save_path'] + \
                    "model_ckpt_" + save_name_base + "_iter_%05i.pt" % (iter + 1))
                torch.save(optimizer.state_dict(), config['ei_training']['save_path'] + \
                    "optimizer_" + save_name_base + ".pt")
                
                if 'time_limit' in config['ei_training'] \
                        and (time() - start_time) / 3600 > config['ei_training']['time_limit']:
                    break


    # Do grid search over parameters
    if config['hmc_grid_search']['do_grid_search']:
        print("Doing grid search over parameters")
        log_step_size_range = [config['hmc_grid_search']['log_step_size_min'],
            config['hmc_grid_search']['log_step_size_max']]
        log_mass_range = [config['hmc_grid_search']['log_mass_min'],
            config['hmc_grid_search']['log_mass_max']]

        for log_step_size in np.linspace(log_step_size_range[0], 
            log_step_size_range[1], num=config['hmc_grid_search']['fidelity']):

            for log_mass in np.linspace(log_mass_range[0], log_mass_range[1],
                num=config['hmc_grid_search']['fidelity']):

                # set the step size and log_mass
                state_dict = flowhmc.state_dict()
                for i in range(flowhmc.end_init_flow_idx, len(flowhmc.flows)-1):
                    state_dict['flows.'+str(i)+'.log_step_size'] = \
                        torch.ones((config['initial_dist_model']['latent_size'],)) * \
                        log_step_size
                    state_dict['flows.'+str(i)+'.log_mass'] = \
                        torch.ones((config['initial_dist_model']['latent_size'],)) * \
                        log_mass
                flowhmc.load_state_dict(state_dict)

                #draw samples
                samples = flowhmc.sample(config['hmc_grid_search']['num_samples'])
                #save samples
                save_name = config['hmc_grid_search']['save_path'] + \
                    'grid_hmc_samples_log_step_size_{:.3f}'.format(log_step_size) + \
                    '_log_mass_{:.3f}'.format(log_mass)
                if config['hmc_grid_search']['include_cl_arg_in_save_name']:
                    suffix = args.processID
                    save_name += '_id_' + suffix
                np.save(save_name, samples.detach().numpy())

    # Calculate the KSD for each setting of the parameters
    if config['hmc_grid_search_ksd_calc']['do_ksd_calc']:
        print("Calculating KSDs for parameter grid")
        # Load dictionary of samples
        # Should have a key of (log_step_size, log_mass) to a value of the samples
        # at those parameter settings
        samples_dict = np.load(config['hmc_grid_search_ksd_calc']['sample_dict_path'],
            allow_pickle=True)
        samples_dict = samples_dict.item()

        # convert samples to internal coords
        for key in samples_dict:
            samples = torch.from_numpy(samples_dict[key])
            ic_samples, _ = flowhmc.flows[-1].inverse(samples)
            samples_dict[key] = ic_samples.detach().numpy()

        # remove any nans
        total_samples = 0
        total_nans = 0
        for key in samples_dict:
            num_samples = samples_dict[key].shape[0]

            samples_dict[key] = \
                samples_dict[key][~np.isnan(samples_dict[key]).any(axis=1)]

            num_nans = num_samples - samples_dict[key].shape[0]
            total_samples += num_samples
            total_nans += num_nans
        print("removed ", total_nans, " nan values.", total_nans/total_samples,
            "proportion of total samples")


        # find a median value to use for all further calculations
        medians = np.array([])
        for key in samples_dict:
            median = bg.utils.get_median_estimate(samples_dict[key])
            medians = np.append(medians, median)
        median = np.median(medians)
        print("overall median", median)

        # convert this into a consistent h_square value
        h_square = 0.5 * median / np.log(samples_dict[next(iter(samples_dict))].shape[0])
        print("h_square value", h_square)

        # compute KSD values
        ksds = {}
        for key in samples_dict:
            samples = samples_dict[key]
            pyt_samples = torch.from_numpy(samples)
            gradlogp = flowhmc.flows[-2].gradlogP(pyt_samples)
            ksds[key] = bg.utils.KSD(samples, gradlogp.detach().numpy(),
                in_h_square=h_square)
            print(key, ": ", ksds[key])

        # save KSD values
        np.save(config['hmc_grid_search_ksd_calc']['ksd_save_path'], ksds) 

    # choose the best hyperparams from a dictionary of param settings vs KSD
    if config['hmc_grid_search_choose_best_hpams']['do_choice']:
        print("Choosing best parameter")
        ksd_dict = np.load(config['hmc_grid_search_choose_best_hpams']['ksd_dict_path'],
            allow_pickle=True)
        ksd_dict = ksd_dict.item()

        # do simple method of choosing the minimum
        min_ksd = 99999
        min_params = ()
        for key in ksd_dict:
            if ksd_dict[key] < min_ksd:
                min_ksd = ksd_dict[key]
                min_params = key

        print("best params: ", min_params)

    # do a general calculation of KSD values for given samples
    if config['general_calc_KSD']['do_general_calc']:
        print("Doing general KSD calculation")
        print("loading samples file",
            config['general_calc_KSD']['samples_path'])
        samples = np.load(config['general_calc_KSD']['samples_path'])

        # convert to internal coords
        pyt_samples = torch.from_numpy(samples)
        ic_samples, _ = flowhmc.flows[-1].inverse(pyt_samples)
        samples = ic_samples.detach().numpy()

        # remove any nans
        num_samples = samples.shape[0]
        samples = samples[~np.isnan(samples).any(axis=1)]
        num_nans = num_samples - samples.shape[0]
        print("removed", num_nans, " nan values.", num_nans/num_samples,
            "proportion of total samples")


        # get gradient values
        pyt_samples = torch.from_numpy(samples)
        gradlogp = flowhmc.flows[-2].gradlogP(pyt_samples).detach().numpy()

        if config['general_calc_KSD']['sub_samples'] is not None:
            num_sub = config['general_calc_KSD']['sub_samples']
            num_samples = samples.shape[0]
            sub_length = math.floor(num_samples/num_sub)
            print("sub_length", sub_length)

            # get a h_square value
            if config['general_calc_KSD']['h_square_val'] is None:
                median = bg.utils.get_median_estimate(samples,
                    num_samples=config['general_calc_KSD']['samples_to_estimate_median'])
                print("median", median)
                h_square = 0.5 * median / np.log(sub_length)
                print("h_square value: ", h_square)
            else:
                h_square = config['general_calc_KSD']['h_square_val']
                print("Using given h_square value of ", h_square)

            KSDs = []
            for i in np.floor(np.linspace(0, num_samples, num=num_sub+1)[0:-1]).astype(int):
                if config['general_calc_KSD']['use_block']:
                    if config['general_calc_KSD']['num_processes'] is None:
                        ksd = bg.utils.blockKSD(samples[i:i+sub_length], gradlogp[i:i+sub_length],
                            config['general_calc_KSD']['num_blocks'], h_square)
                    else:
                        ksd = bg.utils.blockKSDparallel(samples[i:i+sub_length],
                            gradlogp[i:i+sub_length],
                            config['general_calc_KSD']['num_blocks'],
                            h_square,
                            config['general_calc_KSD']['num_processes'])
                    KSDs.append(ksd)
                    print(i, "ksd:", ksd)
                else:
                    ksd = bg.utils.KSD(samples[i:i+sub_length], gradlogp[i:i+sub_length],
                        h_square)
                    KSDs.append(ksd)
                    print(i, "ksd:", ksd)
            KSDs = np.array(KSDs)
            np.save(config['general_calc_KSD']['save_path'], KSDs)
        else:
            # get a h_square value
            if config['general_calc_KSD']['h_square_val'] is None:
                median = bg.utils.get_median_estimate(samples,
                    num_samples=config['general_calc_KSD']['samples_to_estimate_median'])
                print("median", median)
                h_square = 0.5 * median / np.log(samples.shape[0])
                print("h_square value: ", h_square)
            else:
                h_square = config['general_calc_KSD']['h_square_val']
                print("Using given h_square value of ", h_square)

            if config['general_calc_KSD']['use_block']:
                if config['general_calc_KSD']['num_processes'] is None:
                    ksd = bg.utils.blockKSD(samples, gradlogp,
                        config['general_calc_KSD']['num_blocks'], h_square)
                else:
                    ksd = bg.utils.blockKSDparallel(
                        samples,
                        gradlogp,
                        config['general_calc_KSD']['num_blocks'],
                        h_square,
                        config['general_calc_KSD']['num_processes']
                    )
            else:
                ksd = bg.utils.KSD(samples, gradlogp, h_square)
            print("ksd:", ksd)
            np.save(config['general_calc_KSD']['save_path'], np.array([ksd]))

            
    # Just generate some samples and save them
    if config['generate_samples']['do_generation']:
        print("Generating samples")
        for batch_num in range(config['generate_samples']['num_repeats']):
            # Don't track gradients as we are just sampling
            samples = flowhmc.sample(config['generate_samples']['num_samples'],
                require_grad=False)
            save_name = config['generate_samples']['save_path'] + \
                config['generate_samples']['save_name_base'] + \
                '_batch_num_' + str(batch_num)
            if config['generate_samples']['include_cl_arg_in_save_name']:
                save_name += '_processID_' + str(args.processID)
            np.save(save_name, samples.detach().numpy())

    # Estimate KL
    if config['estimate_kl']['do_estimation']:
        print("Estimating KL divergence")
        print("Loading samples file: ", config['estimate_kl']['samples_file'])
        samples = np.load(config['estimate_kl']['samples_file'])

        if config['estimate_kl']['num_samples'] is not None:
            samples = samples[0:config['estimate_kl']['num_samples'], :]

        # convert to internal coords
        pyt_samples = torch.from_numpy(samples)
        ic_samples, _ = flowhmc.flows[-1].inverse(pyt_samples)
        samples = ic_samples.detach().numpy()

        ic_training_data, _ = flowhmc.flows[-1].inverse(flowhmc.training_data)
        ic_training_data = ic_training_data.detach().numpy()

        # remove any nans
        num_samples = samples.shape[0]
        samples = samples[~np.isnan(samples).any(axis=1)]
        num_nans = num_samples - samples.shape[0]
        print("removed", num_nans, " nan values.", num_nans/num_samples,
            "proportion of total samples")

        cart_indices = [x for x in range(9)]
        bond_indices = [3*x+9 for x in range(17)]
        angle_indices = [3*x+10 for x in range(17)]
        dihedral_indices = [3*x+11 for x in range(17)]

        rangeminmax = 7

        def kl(samples, training_data):
            if config['estimate_kl']['KL_direction'] == 'forward':
                return bg.utils.estimate_kl(training_data, samples, -rangeminmax,
                    rangeminmax)
            elif config['estimate_kl']['KL_direction'] == 'reverse':
                return bg.utils.estimate_kl(samples, training_data, -rangeminmax,
                    rangeminmax)
            else:
                print("Given an unknown KL direction in the config file:",
                    config['estimate_kl']['KL_direction'])

        kls = []
        print("Computing KLs")
        for i in tqdm(range(60)):
            kls.append(kl(samples[:, i], ic_training_data[:, i]))
        kls = np.array(kls)

        print("Saving KLs at", config['estimate_kl']['save_name'])
        np.savetxt(config['estimate_kl']['save_name'], kls)

        cart_kls = kls[cart_indices]
        bond_kls = kls[bond_indices]
        angle_kls = kls[angle_indices]
        dihedral_kls = kls[dihedral_indices]

        def print_stats(in_kls):
            print("Mean", np.mean(in_kls))
            print("Median", np.median(in_kls))
            print("Min", np.min(in_kls))
            print("Max", np.max(in_kls))


        print("Cartesian group statistics:")
        print_stats(cart_kls)
        print("Bond group statistics:")
        print_stats(bond_kls)
        print("Angle group statistics:")
        print_stats(angle_kls)
        print("Dihedral group statistics:")
        print_stats(dihedral_kls)

    # Compute KL values for a grid of samples from different parameter values
    if config['grid_search_kl_calc']['do_kl_calc']:
        print("Finding KLs over grid of samples")
        print("Loading sample dict file",
            config['grid_search_kl_calc']['sample_dict_path'])
        samples_dict = np.load(
            config['grid_search_kl_calc']['sample_dict_path'],
            allow_pickle=True)
        samples_dict = samples_dict.item()

        # convert samples to internal coords
        for key in samples_dict:
            samples = torch.from_numpy(samples_dict[key])
            ic_samples, _ = flowhmc.flows[-1].inverse(samples)
            samples_dict[key] = ic_samples.detach().numpy()

        # remove any nans
        total_samples = 0
        total_nans = 0
        for key in samples_dict:
            num_samples = samples_dict[key].shape[0]

            samples_dict[key] = \
                samples_dict[key][~np.isnan(samples_dict[key]).any(axis=1)]

            num_nans = num_samples - samples_dict[key].shape[0]
            total_samples += num_samples
            total_nans += num_nans
        print("removed ", total_nans, " nan values.", total_nans/total_samples,
            "proportion of total samples")

        ic_training_data, _ = flowhmc.flows[-1].inverse(flowhmc.training_data)
        ic_training_data = ic_training_data.detach().numpy()

        rangeminmax = 7

        def kl(samples, training_data):
            if config['grid_search_kl_calc']['KL_direction'] == 'forward':
                return bg.utils.estimate_kl(training_data, samples, -rangeminmax,
                    rangeminmax)
            elif config['grid_search_kl_calc']['KL_direction'] == 'reverse':
                return bg.utils.estimate_kl(samples, training_data, -rangeminmax,
                    rangeminmax)
            else:
                print("Given an unknown KL direction in the config file:",
                    config['grid_search_kl_calc']['KL_direction'])

        kls_dict = {}
        print("Computing KLs")
        for key in tqdm(samples_dict):
            kls = []
            for i in range(60):
                kls.append(kl(samples_dict[key][:, i], ic_training_data[:, i]))
            kls = np.array(kls)
            print(kls)
            kls_dict[key] = kls

        print("Saving KL dict at", 
            config['grid_search_kl_calc']['kls_save_path'])
        np.save(config['grid_search_kl_calc']['kls_save_path'], kls_dict)

    if config['eval_log_target']['do_eval']:
        print("Evaluating log target at samples",
            config['eval_log_target']['samples'])
        samples = np.load(config['eval_log_target']['samples'])

        # convert to internal coords
        pyt_samples = torch.from_numpy(samples)
        samples, _ = flowhmc.flows[-1].inverse(pyt_samples)

        # remove any nans
        num_samples = samples.shape[0]
        samples = samples[~torch.isnan(samples).any(axis=1)]
        num_nans = num_samples - samples.shape[0]
        print("removed", num_nans, " nan values.", num_nans/num_samples,
            "proportion of total samples")

        if config['eval_log_target']['num_split'] is not None:
            indeces = np.floor(np.linspace(0, samples.shape[0],
                num=config['eval_log_target']['num_split']+1)).astype(int)
            mean_log_targets = []
            for i in range(len(indeces)-1):
                sub_samples = samples[indeces[i]:indeces[i+1],:]
                log_targets = flowhmc.flows[-2].target.log_prob(sub_samples)
                mean_log_targets.append(np.mean(log_targets.detach().numpy()))
                print("Mean log target", mean_log_targets[-1])
            mean_log_targets = np.array(mean_log_targets)
            np.savetxt(config['eval_log_target']['save_path'], mean_log_targets)
        else:
            log_targets = flowhmc.flows[-2].target.log_prob(samples)
            mean_log_target = np.mean(log_targets.detach().numpy())
            print("Mean log target", mean_log_target)
            np.savetxt(config['eval_log_target']['save_path'],
                np.array([mean_log_target]))

    if config['estimate_sksd']['do_estimation']:
        print("Estimating SKSD with samples",
            config['estimate_sksd']['samples'])

        samples = np.load(config['estimate_sksd']['samples'])

        g = torch.eye(60).double()


        # convert to internal coords
        print("Converting to internal coords")
        pyt_samples = torch.from_numpy(samples).double()
        ic_samples, _ = flowhmc.flows[-1].inverse(pyt_samples)

        # Median estimation
        proj_x = torch.matmul(ic_samples, g.transpose(0,1)) # (N x dim)
        transpose_proj_x = torch.transpose(proj_x, 0, 1)
        exp_transpose_proj_x = torch.unsqueeze(transpose_proj_x, 2) # (dim x N x 1)
        exp_transpose_proj_x = exp_transpose_proj_x.contiguous()
        num_median = config['estimate_sksd']['num_median']
        squared_pairwise_distances = torch.cdist(
            exp_transpose_proj_x[:, 0:num_median, :],
            exp_transpose_proj_x[:, 0:num_median, :]) ** 2
        median_squared_distances = torch.median(
            torch.flatten(squared_pairwise_distances, start_dim=1, end_dim=2),
            dim=1)[0]

        # get gradient values
        print("Getting gradients")
        gradlogp = flowhmc.flows[-2].gradlogP(ic_samples)


        print("Calculating SKSDs")
        num_samples = config['estimate_sksd']['num_samples_in_batch']
        num_batches = config['estimate_sksd']['num_batches']
        num_blocks = config['estimate_sksd']['num_blocks']
        sksds = []
        for i in range(num_batches):
            sub_samples = ic_samples[i*num_samples:(i+1)*num_samples,:]
            sub_grads = gradlogp[i*num_samples:(i+1)*num_samples,:]
            sksd = bg.utils.blockSKSD(sub_samples, sub_grads, g, num_blocks,
                input_median=median_squared_distances)
            print(i, sksd)
            sksds.append(sksd.detach().numpy())
        sksds = np.array(sksds)
        np.savetxt(config['estimate_sksd']['save_name'], sksds)


    if config['train_ei_sksd']['do_train']:
        print("Training with EI and SKSD")
        hmc_optimizer = torch.optim.Adam(flowhmc.get_hmc_parameters(),
            lr=config['train_ei_sksd']['ei_lr'])

        if config['initial_dist_model']['scaling'] is not None:
            scale_optimizer = torch.optim.Adam(
                [flowhmc.flows[flowhmc.end_init_flow_idx-1].log_scale],
                lr=config['train_ei_sksd']['sksd_lr']
            )
        elif config['initial_dist_model']['noise_std'] is not None:
            scale_optimizer = torch.optim.Adam(
                [flowhmc.flows[flowhmc.end_init_flow_idx-1].log_std],
                lr=config['train_ei_sksd']['sksd_lr']
            )
        else:
            raise ValueError("""Training with EI and SKSD requires either the 
                                scaling or noise_std config parameter to be
                                set""")

        ei_losses = np.array([])
        sksd_losses = np.array([])
        scales = np.array([])

        save_name_base = "hmc_ei_sksd"

        if config['train_ei_sksd']['resume']:
            latest_cp = bg.utils.get_latest_checkpoint(config['train_ei_sksd']['save_path'], 'model')
            if latest_cp is not None:
                start_iter = int(latest_cp[-8:-3])
                flowhmc.load_state_dict(torch.load(latest_cp))
                hmc_optimizer_path = config['train_ei_sksd']['save_path'] + \
                                     "hmc_optimizer_" + save_name_base + ".pt"
                if os.path.exists(hmc_optimizer_path):
                    hmc_optimizer.load_state_dict(torch.load(hmc_optimizer_path))
                scale_optimizer_path = config['train_ei_sksd']['save_path'] + \
                                       "scale_optimizer_" + save_name_base + ".pt"
                if os.path.exists(scale_optimizer_path):
                    scale_optimizer.load_state_dict(torch.load(scale_optimizer_path))
                trainprog_path = config['train_ei_sksd']['save_path'] + "trainprog_" + \
                                 save_name_base + "_ckpt_%05i.txt" % start_iter
                if os.path.exists(trainprog_path):
                    data = np.loadtxt(trainprog_path)
                    ei_losses = data[:, 0]
                    sksd_losses = data[:, 1]
                    scales = data[:, 2]
            else:
                start_iter = 0
        else:
            start_iter = 0

        if config['train_ei_sksd']['sksd_lr_decay_factor'] is not None:
            do_decay = True
            decay_schedule = torch.optim.lr_scheduler.ExponentialLR(
                scale_optimizer,
                config['train_ei_sksd']['sksd_lr_decay_factor'])
            decay_steps = config['train_ei_sksd']['sksd_lr_decay_steps']
            if start_iter > 0:
                for _ in range(start_iter // decay_steps):
                    decay_schedule.step()
        else:
            do_decay = False

        start_time = time()

        print("Iter    EI loss     SKSD      scale")
        for iter in range(start_iter, config['train_ei_sksd']['iters']):
            hmc_optimizer.zero_grad()

            x, _ = flowhmc.rnvp_initial_dist.forward(
                config['train_ei_sksd']['samples_per_iter'])
            for i in range(len(flowhmc.flows)-1):
                x, _ = flowhmc.flows[i].forward(x)
            
            log_probs = flowhmc.target_dist.log_prob(x)
            ei_loss = -torch.mean(log_probs)

            if not torch.isnan(ei_loss):
                ei_loss.backward(retain_graph=True)
            hmc_optimizer.step()

            scale_optimizer.zero_grad()
            gradlogp = flowhmc.flows[-2].gradlogP(x)
            g = torch.eye(60).double()
            sksd = bg.utils.SKSD(x, gradlogp, g)
            sksd.backward()
            scale_optimizer.step()


            if config['initial_dist_model']['scaling'] is not None:
                scale = torch.exp(flowhmc.flows[flowhmc.end_init_flow_idx-1].log_scale)
            elif config['initial_dist_model']['noise_std'] is not None:
                scale = torch.exp(flowhmc.flows[flowhmc.end_init_flow_idx-1].log_std)

            print("{:4}".format(iter),
                "{:10.4f}".format(ei_loss),
                "{:10.4f}".format(sksd),
                "{:10.4f}".format(scale))

            ei_losses = np.append(ei_losses, ei_loss.detach().numpy())
            sksd_losses = np.append(sksd_losses, sksd.detach().numpy())
            scales = np.append(scales, scale.detach().numpy())

            if do_decay:
                if iter%decay_steps == 0:
                    decay_schedule.step()

            if (iter + 1) % config['train_ei_sksd']['save_interval'] == 0:
                data = np.column_stack((ei_losses, sksd_losses, scales))
                np.savetxt(config['train_ei_sksd']['save_path'] + "trainprog_" + \
                    save_name_base + "_ckpt_%05i.txt" % (iter + 1), data)
                torch.save(flowhmc.state_dict(), config['train_ei_sksd']['save_path'] + \
                    "model_ckpt_" + save_name_base + "_iter_%05i.pt" % (iter + 1))
                torch.save(hmc_optimizer.state_dict(), config['train_ei_sksd']['save_path'] + \
                           "hmc_optimizer_" + save_name_base + ".pt")
                torch.save(scale_optimizer.state_dict(), config['train_ei_sksd']['save_path'] + \
                           "scale_optimizer_" + save_name_base + ".pt")

                if 'time_limit' in config['train_ei_sksd'] \
                        and (time() - start_time) / 3600 > config['train_ei_sksd']['time_limit']:
                    break




if __name__ == "__main__":
    main()


# %%
