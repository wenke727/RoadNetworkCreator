import random
import geopandas as gpd
from shapely.geometry import point
from tqdm import tqdm

from pano_base import query_pano
from pano_img import get_staticimage, traverse_panos_by_rid, PANO_log, pano_dir

from utils.coord.coord_transfer import wgs_to_bd_mc, bd_coord_to_mc

status, pano_respond, panos, nxt = query_pano( 12681303.05, 2582504.44, visualize=True, add_to_DB=False )
panos.iloc[1:-1].sample(1)


roads = gpd.read_file('./shanghai_test_pano.geojson')
points = roads.geometry.apply(lambda x: x.coords[1]).values.tolist()

random.seed(1)

def get_pano_img_by_coord(lon, lat, coord='wgs'):
    """

    Args:
        lon ([type]): [description]
        lat ([type]): [description]

    Returns:
        [type]: [description]
    """
    if coord =='wgs':
        x, y = wgs_to_bd_mc(lon, lat)
    else:
        x, y = bd_coord_to_mc(lon, lat)
    status, pano_respond, panos, nxt = query_pano( x, y, visualize=False, add_to_DB=False )

    if panos is None or panos.shape[0] < 3:
        return None
    
    id = random.randint( 1, panos.shape[0]-2 )
    pano = panos.iloc[id]
    # fn = f"{pano_dir}/{pano.RID}_{pano.Order:02d}_{pano.PID}_{pano.DIR}.jpg"
    fn = f"./tmp/{pano.RID}_{pano.Order:02d}_{pano.PID}_{pano.DIR}.jpg"
    fn = get_staticimage( pano.PID, pano.DIR, fn, PANO_log )
    
    return fn

# get_pano_img_by_coord(116.403411,39.914001)

res = []
for lon, lat in tqdm(points[145:]):
    fn = get_pano_img_by_coord(lon, lat)
    if fn is not None:
        res.append( fn )


