import boltzgen.mixed as mixed
import normflow as nf

class CoordinateTransform(nf.flows.Flow):
    """
    Coordinate transform for Boltzmann generators, see
    https://science.sciencemag.org/content/365/6457/eaaw1147
    """
    def __init__(self, data, n_dim, z_matrix, backbone_indices):
        """
        Constructor
        :param data: Data used to initialize transformation
        :param n_dim: Number of dimensions in original space
        :param z_matrix: Defines which atoms to represent in internal coordinates
        :param backbone_indices: Indices of atoms of backbone, will be left in
        cartesian coordinates
        """

        self.mixed_transform = mixed.MixedTransform(n_dim, backbone_indices, z_matrix, data)

    def forward(self, z):
        z_, log_det = self.mixed_transform.forward(z)
        return z_, log_det

    def inverse(self, z):
        z_, log_det = self.mixed_transform.inverse(z)
        return z_, log_det
