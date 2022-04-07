from . import mixed
from . import internal
import normflow as nf
import torch



class CoordinateTransform(nf.flows.Flow):
    """
    Coordinate transform for Boltzmann generators, see
    https://science.sciencemag.org/content/365/6457/eaaw1147

    The code of this function was taken from
    https://github.com/maccallumlab/BoltzmannGenerator

    Meaning of forward and backward pass are switched to meet
    convention of normflow package
    """
    def __init__(self, data, n_dim, z_matrix, backbone_indices, mode='mixed'):
        """
        Constructor
        :param data: Data used to initialize transformation
        :param n_dim: Number of dimensions in original space
        :param z_matrix: Defines which atoms to represent in internal coordinates
        :param backbone_indices: Indices of atoms of backbone, will be left in
        cartesian coordinates or are the last to be converted to internal coordinates
        :param mode: Mode of the coordinate transform, can be mixed or internal
        """
        super().__init__()
        if mode == 'mixed':
            self.transform = mixed.MixedTransform(n_dim, backbone_indices, z_matrix, data)
        elif mode == 'internal':
            self.transform = internal.CompleteInternalCoordinateTransform(n_dim, z_matrix,
                                                                          backbone_indices, data)
        else:
            raise NotImplementedError('This mode is not implemented.')

    def forward(self, z):
        z_, log_det = self.transform.inverse(z)
        return z_, log_det

    def inverse(self, z):
        z_, log_det = self.transform.forward(z)
        return z_, log_det

class Scaling(nf.flows.Flow):
    """
    Applys a scaling factor
    """
    def __init__(self, mean, log_scale):
        """
        Constructor
        :param means: The mean of the previous layer
        :param log_scale: The log of the scale factor to apply
        """
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_parameter('log_scale', torch.nn.Parameter(log_scale))

    def forward(self, z):
        scale = torch.exp(self.log_scale)
        z_ = (z-self.mean) * scale + self.mean
        logdet = torch.log(scale) * self.mean.shape[0]
        return z_, logdet

    def inverse(self, z):
        scale = torch.exp(self.log_scale)
        z_ = (z-self.mean) / scale + self.mean
        logdet = -torch.log(scale) * self.mean.shape[0]
        return z_, logdet

class AddNoise(nf.flows.Flow):
    """
    Adds a small amount of Gaussian noise
    """
    def __init__(self, log_std):
        """
        Constructor
        :param log_std: The log standard deviation of the noise
        """
        super().__init__()
        self.register_parameter('log_std', torch.nn.Parameter(log_std))

    def forward(self, z):
        eps = torch.randn_like(z)
        z_ = z + torch.exp(self.log_std) * eps
        logdet = torch.zeros(z_.shape[0])
        return z_, logdet

    def inverse(self, z):
        return self.forward(z)
