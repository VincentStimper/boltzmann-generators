from simtk import openmm as mm
from simtk.openmm import app
from simtk import unit
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
        energies = torch.zeros((n_batch, 1), dtype=input.dtype)
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
                    state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule / unit.mole) / kBT
                )

                # get forces
                f = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule / unit.mole / unit.nanometer
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
    def var_init(sys, temp):
        """
        Method to initialize temperature and openmm context for workers
        of multiprocessing pool
        """
        global temperature, openmm_context
        temperature = temp
        sim = app.Simulation(sys.topology, sys.system,
                             mm.LangevinIntegrator(temp * unit.kelvin,
                                                   1.0 / unit.picosecond,
                                                   1.0 * unit.femtosecond),
                             platform=mm.Platform.getPlatformByName('Reference'))
        openmm_context = sim.context

    @staticmethod
    def batch_proc(input):
        # Process state
        # openmm context and temperature are passed a global variables
        input = input.reshape(-1, 3)
        n_dim = input.shape[0]

        kBT = R * temperature
        # Handle nans and infinities
        if np.any(np.isnan(input)) or np.any(np.isinf(input)):
            energy = np.nan
            force = np.zeros_like(input)
        else:
            openmm_context.setPositions(input)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energy = state.getPotentialEnergy().value_in_unit(
                unit.kilojoule / unit.mole) / kBT

            # get forces
            force = -state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule / unit.mole / unit.nanometer) / kBT
        force = force.reshape(n_dim * 3)
        return energy, force

    @staticmethod
    def forward(ctx, input, pool):
        device = input.device
        input_np = input.cpu().detach().numpy()
        energies_out, forces_out = zip(*pool.map(
            OpenMMEnergyInterfaceParallel.batch_proc, input_np))
        energies_np = np.array(energies_out)[:, None]
        forces_np = np.array(forces_out)
        energies = torch.from_numpy(energies_np)
        forces = torch.from_numpy(forces_np)
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
