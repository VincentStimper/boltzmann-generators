import torch
import yaml

def get_config(file_path):
    """
    Read configuration parameter form file
    :param file_path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(file_path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def get_coord(traj):
    """
    Get coordinates from trajectory as torch tensor
    :param traj: Openmm trajectory
    :return: Torch tensor with coordinates
    """

    traj.center_coordinates()

    # superpose on the backbone
    ind = traj.top.select("backbone")

    traj.superpose(traj, 0, atom_indices=ind, ref_atom_indices=ind)

    # Gather the training data into a pytorch Tensor with the right shape
    coord_np = traj.xyz
    n_atoms = coord_np.shape[1]
    n_dim = n_atoms * 3
    coord_np = coord_np.reshape(-1, n_dim)
    coord = torch.from_numpy(coord_np.astype("float64"))

    return coord