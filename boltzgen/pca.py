import torch
import torch.nn as nn
import math


class PCATransform(nn.Module):
    def __init__(self, dims_in, training_data, drop_dims=6):
        super().__init__()
        self.dims = dims_in
        assert drop_dims >= 0, "drop_dims must be non-negative"
        assert dims_in - drop_dims > 0, "dims_in - drop_dims must be positive"
        self.drop_dims = drop_dims

        # compute our whiten / blackening matrices, etc
        training_data = torch.as_tensor(training_data)
        self._validate_training_data(training_data)
        self._compute_decomp(training_data)

    def forward(self, x, context=None):
        x = x.unsqueeze(2)
        x = x - self.means.unsqueeze(1).expand_as(x)
        x = torch.matmul(self.whiten, x).squeeze(2)
        return x, self.jac.expand(x.shape[0])

    def inverse(self, x, context=None):
        x = x.unsqueeze(2)
        x = torch.matmul(self.blacken, x).squeeze(2)
        x = x + self.means.unsqueeze(0).expand_as(x)
        return x, -self.jac.expand(x.shape[0])

    def _validate_training_data(self, training_data):
        if training_data is None:
            raise ValueError("PCA must be supplied with training_data.")

        if len(training_data.shape) != 2:
            raise ValueError("training_data must be n_samples x n_dim array")

        n_samp = training_data.shape[0]
        n_dim = training_data.shape[1]

        if n_dim != self.dims:
            raise ValueError(
                f"training_data must have {self.dims} dimensions, not {n_dim}."
            )

        if not n_samp >= n_dim:
            raise ValueError("training_data must have n_samp >= n_dim")

    def _compute_decomp(self, training_data):
        with torch.no_grad():
            # These will be the dimensions we keep
            keep_dims = self.dims - self.drop_dims

            # mean center the data
            means = torch.mean(training_data, dim=0)
            self.register_buffer("means", means)
            training_data = training_data - self.means.expand_as(training_data)

            # do the SVD
            U, S, V = torch.svd(training_data)

            # All eigenvalues should be positive.
            if torch.any(S[:keep_dims] <= 0):
                raise RuntimeError("All non-dropped eigenvalues should be positive.")

            # Compute the standard deviation and throw away any dimensions we don't
            # want to keep
            stds = S[:keep_dims] / math.sqrt(training_data.shape[0] - 1)
            self.register_buffer("stds", stds)
            V = V[:, :keep_dims]

            # Store the jacobian for later.
            jac = -torch.sum(torch.log(self.stds))
            self.register_buffer("jac", jac)

            # Store the whitening / blackening matrices for later.
            # The unsqueeze(0) adds a dummy leading dimension, which
            # allows us to matrix multiply by a batch of samples.
            whiten = (torch.diag(1.0 / self.stds) @ V.t()).unsqueeze(0)
            blacken = (V @ torch.diag(self.stds)).unsqueeze(0)
            self.register_buffer("whiten", whiten)
            self.register_buffer("blacken", blacken)

