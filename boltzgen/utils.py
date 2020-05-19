import yaml

import torch
import mdtraj

def get_config(path):
    """
    Read configuration parameter form file
    :param path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)


def get_coord(path):
    """
    Get coordinates from h5 trajectory file as torch tensor
    :param file_path: String, path to h5 trajectory file
    :return: Torch tensor with coordinates
    """

    # Load trajectory
    traj = mdtraj.load(path)

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