import yaml
import os

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


def load_traj(path):
    """
    Load coordinates from h5 trajectory file as torch tensor
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


def get_latest_checkpoint(dir_path, key=''):
    """
    Get path to latest checkpoint in directory
    :param dir_path: Path to directory to search for checkpoints
    :param key: Key which has to be in checkpoint name
    :return: Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f and ".pt" in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]