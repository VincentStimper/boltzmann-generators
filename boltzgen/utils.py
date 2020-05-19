import yaml

def get_config(file_path):
    """
    Read configuration parameter form file
    :param file_path: Path to the yaml configuration file
    :return: Dict with parameter
    """

    with open(file_path, 'r') as stream:
        return yaml.load(stream)