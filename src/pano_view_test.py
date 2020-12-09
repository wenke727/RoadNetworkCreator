from db_process import load_from_DB, store_to_DB, ENGINE
import urllib
import os
import yaml
from PIL import Image
import time
import random 
import geopandas as gpd
from tqdm import tqdm

from mapAPI import get_staticimage
from roadNetwork import map_visualize

with open(os.path.join( os.path.dirname(__file__), 'config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

pano_dir = config['data']['pano_dir']


DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
# store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)


def query_static_imgs_by_road(name = '光侨路'):
    rids = DB_roads.query(f"Name == '{name}' ").RID.unique().tolist()
    df = DB_panos.query( f"RID in {rids}" )
    DB_panos.groupby('RID').count()

    for index, item in tqdm( df[130:].iterrows()):
        if item.DIR == 0:
            continue
        
        if get_staticimage(item.PID, heading=item.DIR):
            time.sleep(random.triangular(0.5, 1, 10))

    return True




DB_pano_base
DB_panos
DB_connectors
DB_roads


# query the connect lane
DB_roads.query( 'PID_start == PID_end' )

res = []
pid = ['09005700121902131650266199U']

while len(pid) > 0:
    pid_nxt = DB_roads.query( f"PID_start in {pid}" ).PID_end.values.tolist()
    res += pid
    print(len(pid))




DB_roads.query( f"PID_start in {pid_nxt}" )

DB_connectors.query(f'prev_pano_id in {pid_nxt}')



