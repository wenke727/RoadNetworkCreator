import yaml
import os


def load_config(fn=None):
    if fn is None:
        fn = os.path.join( os.path.dirname(__file__), '../config.yaml')
        
    with open(fn) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


if __name__ == "__main__":
    config = load_config()
    pass