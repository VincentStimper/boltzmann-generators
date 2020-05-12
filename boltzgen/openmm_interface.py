from simtk import openmm as mm
from simtk.openmm import app
from simtk.unit import kelvin, kilojoule, mole, nanometer
import torch
import numpy as np


# Gas constant in kJ / mol / K
R = 8.314e-3


class OpenMMEnergyInterface(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, openmm_context, temperature):
        device = input.device
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1))
        forces = torch.zeros_like(input)

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = (
                    state.getPotentialEnergy().value_in_unit(kilojoule / mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        kilojoule / mole / nanometer
                    )
                    / kBT
                )
                forces[i, :] = torch.from_numpy(-f)
        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


class OpenMMEnergyInterfaceParallel(torch.autograd.Function):
    """
    Uses parallel processing to get the energies of the batch of states
    """
    @staticmethod
    def batch_proc(input):
        # Process batch  of states
        # openmm context and temperature are passed a global variables
        n_batch = input.shape[0]
        input = input.view(n_batch, -1, 3)
        n_dim = input.shape[1]
        energies = torch.zeros((n_batch, 1))
        forces = torch.zeros((n_batch, n_dim, 3))

        kBT = R * temp
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            # Handle nans and infinities
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                energies[i, 0] = np.nan
            else:
                openmm_context.setPositions(x)
                state = openmm_context.getState(getForces=True, getEnergy=True)

                # get energy
                energies[i, 0] = state.getPotentialEnergy().value_in_unit(kilojoule / mole) / kBT

                # get forces
                f = state.getForces(asNumpy=True).value_in_unit(
                    kilojoule / mole / nanometer) / kBT
                forces[i, :] = torch.from_numpy(-f)
        forces = forces.view(n_batch, n_dim * 3)
        return energies, forces

    @staticmethod
    def forward(ctx, input, pool, split_length):
        device = input.device
        input_splitted = torch.split(input, split_length)
        energies_, forces_ = zip(*pool.map(OpenMMEnergyInterfaceParallel.batch_proc,
                                           input_splitted))
        energies = torch.cat(energies_)
        forces = torch.cat(forces_)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


def regularize_energy(energy, energy_cut, energy_max):
    # Cast inputs to same type
    energy_cut = energy_cut.type(energy.type())
    energy_max = energy_max.type(energy.type())
    # Check whether energy finite
    energy_finite = torch.isfinite(energy)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut
    )
    energy = torch.where(energy_finite, energy,
                         torch.tensor(np.nan, dtype=energy.dtype, device=energy.device))
    return energy
