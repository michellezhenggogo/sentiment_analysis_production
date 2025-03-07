import yaml
import os


def load_config():
    """
    Load YAML config file and return as a dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Load the config globally
CONFIG = load_config()
