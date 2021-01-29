import yaml
import os


def load_config(fn=None):
    if fn is None:
        fn = os.path.join( os.path.dirname(__file__), '../config.yaml')
        
    with open(fn) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def clear_folder(folder = '/home/pcl/Data/minio_server/panos_data/Futian/益田路'):
    os.popen( f"cd {folder}; rm *" )
    return 


if __name__ == "__main__":
    config = load_config()
    clear_folder( "../../../data/label_data" )
    pass