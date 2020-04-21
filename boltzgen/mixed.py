from .pca import PCATransform
from .internal import InternalCoordinateTransform
from . import zmatrix
import torch
import torch.nn as nn
import math
import numpy as np
from collections import namedtuple
import itertools


class MixedTransform(nn.Module):
    def __init__(
        self,
        n_dim,
        cartesian_indices,
        z_mat,
        training_data,
    ):
        super().__init__()
        # cartesian indices are the atom indices of the atoms that are not
        # represented in internal coordinates but are left as cartesian
        # e.g. for 22 atoms it could be [4, 5, 6, 8, 14, 15, 16, 18]
        self.n_dim = n_dim
        self.len_cart_inds = len(cartesian_indices)

        # Create our internal coordinate transform
        self.ic_transform = InternalCoordinateTransform(
            n_dim, z_mat, cartesian_indices, training_data
        )

        # permute puts the cartesian coords first then the internal ones
        # permute_inv does the opposite
        permute = torch.zeros(n_dim, dtype=torch.long)
        permute_inv = torch.zeros(n_dim, dtype=torch.long)
        all_ind = cartesian_indices + [row[0] for row in z_mat]
        for i, j in enumerate(all_ind):
            permute[3 * i + 0] = torch.as_tensor(3 * j + 0, dtype=torch.long)
            permute[3 * i + 1] = torch.as_tensor(3 * j + 1, dtype=torch.long)
            permute[3 * i + 2] = torch.as_tensor(3 * j + 2, dtype=torch.long)
            permute_inv[3 * j + 0] = torch.as_tensor(3 * i + 0, dtype=torch.long)
            permute_inv[3 * j + 1] = torch.as_tensor(3 * i + 1, dtype=torch.long)
            permute_inv[3 * j + 2] = torch.as_tensor(3 * i + 2, dtype=torch.long)
        self.register_buffer("permute", permute)
        self.register_buffer("permute_inv", permute_inv)


        training_data = training_data[:, self.permute]

        self.pca_transform = PCATransform(
            dims_in=3*len(cartesian_indices),
            training_data=training_data[:, range(3*len(cartesian_indices))]
            )

    def forward(self, x):

        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])
        print(x.size())

        # Run transform to internal coordinates.
        x, new_jac = self.ic_transform.forward(x)
        jac = jac + new_jac
        print(x.size())

        # Permute to put PCAs first.
        x = x[:, self.permute]
        print(x.size())

        # Split off the PCA coordinates and internal coordinates
        pca_input = x[:, :3*self.len_cart_inds]
        int_coords = x[:, 3*self.len_cart_inds:]

        # Run through PCA.
        pca_output, pca_jac = self.pca_transform(pca_input)
        jac = jac + pca_jac

        # Merge everything back together.
        x = torch.cat([pca_output] + [int_coords], dim=1)
        print(x.size())

        return x, jac

    def inverse(self, x):
        # Create the jacobian vector
        jac = x.new_zeros(x.shape[0])

        # Separate out the PCAs and internal coordinates
        # pca removes 6 dims because of the rototranslationally invariant system
        pca_input = x[:, :3*self.len_cart_inds-6]
        int_coords = x[:, 3*self.len_cart_inds-6:]

        # Run through PCA
        pca_output, pca_jac = self.pca_transform.inverse(pca_input)
        jac = jac + pca_jac

        # Merge everything back together
        x = torch.cat([pca_output] + [int_coords], dim=1)

        # Permute back into atom order
        x = x[:, self.permute_inv]

        # Run through inverse internal coordinate transform
        x, new_jac = self.ic_transform.inverse(x)
        jac = jac + new_jac

        return x, jac

    #debugging
    def print_tensor(self, x):
        for i in range(int(x.shape[1]/3)):
            print(x[0, 3*i:3*i+3])
