from simtk import openmm as mm
from simtk.openmm import app
from simtk.unit import kelvin, kilojoule, mole, nanometer
import torch


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
        forces = torch.zeros((n_batch, n_dim, 3))

        kBT = R * temperature
        input = input.cpu().detach().numpy()
        for i in range(n_batch):
            # reshape the coordinates and send to OpenMM
            x = input[i, :].reshape(-1, 3)
            openmm_context.setPositions(x)
            state = openmm_context.getState(getForces=True, getEnergy=True)

            # get energy
            energies[i] = (
                state.getPotentialEnergy().value_in_unit(kilojoule / mole) / kBT
            )

            # get forces
            f = (
                state.getForces(asNumpy=True).value_in_unit(
                    kilojoule / mole / nanometer
                )
                / kBT
            )
            forces[i, :] = torch.from_numpy(-f.astype("float32"))
        forces = forces.view(n_batch, n_dim * 3)
        # Save the forces for the backward step, uploading to the gpu if needed
        ctx.save_for_backward(forces.to(device=device))
        return energies.to(device=device)

    @staticmethod
    def backward(ctx, grad_output):
        forces, = ctx.saved_tensors
        return forces * grad_output, None, None


openmm_energy = OpenMMEnergyInterface.apply


def regularize_energy(energy, energy_cut, energy_max):
    # Fill any NaNs with energy_max
    energy = torch.where(torch.isfinite(energy), energy, energy_max)
    # Cap the energy at energy_max
    energy = torch.where(energy < energy_max, energy, energy_max)
    # Make it logarithmic above energy cut and linear below
    energy = torch.where(
        energy < energy_cut, energy, torch.log(energy - energy_cut + 1) + energy_cut
    )
    return energy
