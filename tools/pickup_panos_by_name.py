import os
import shutil
from zipfile import ZipFile
from tqdm import tqdm

PANO_DIR = '/home/pcl/Data/minio_server/panos'
folder = '/home/pcl/Data/pano_dataSet'

for fs in os.listdir(folder):
    if 'zip' not in fs: 
        continue
    
    # sub_folder = fs
    fn = os.path.join(folder, fs)
    sub_folder = os.path.join(folder, fn.split('.')[0])

    with ZipFile(fn, 'r') as f:
        names = ["_".join(info.filename.split('/')[-1].split("_")[-4:]) for info in f.infolist()]

    os.popen( f"mkdir {sub_folder}" )
    for name in tqdm(names):
        if 'jpg' not in name: continue
        fn = os.path.join(PANO_DIR, name)
        shutil.copy( fn, sub_folder )



# os.popen()