import os, sys
import random
import pinyin
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import point

sys.path.append("/home/pcl/traffic/RoadNetworkCreator_by_View/src")
from pano_base import query_pano
from pano_img import fetch_pano_img, traverse_panos_by_rid, PANO_log, pano_dir
from utils.coord.coord_transfer import wgs_to_bd_mc, bd_coord_to_mc


def get_pano_img_by_coord(lon, lat, folder='./shanghai', coord='wgs'):
    """
    Args:
        lon ([type]): [description]
        lat ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
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
    fn = f"{folder}/{pano.RID}_{pano.Order:02d}_{pano.PID}_{pano.DIR}.jpg"
    fn = fetch_pano_img( pano.PID, pano.DIR, fn, logger=PANO_log )
    
    return fn


def start():
    random.seed(1)

    citis = ['北京','上海', '广州', '杭州']
    # create_test_dataset_for_cites(citis)
    
    for city in citis[3:]:
        print(city)
        city_pinyin= pinyin.get(city, format='strip')
        fn = f'/home/pcl/Data/minio_server/input/{city_pinyin}_test_pano.geojson'

        # roads = gpd.read_file(fn).sample(500, random_state=1)
        roads = gpd.read_file(fn)
        points = roads.geometry.apply(lambda x: x.coords[1] if len(x.coords) < 3 else x.coords[int(len(x.coords)//2)] ).values.tolist()

        res, error_lst = [], []
        for lon, lat in tqdm(points):
            try:
                fn = get_pano_img_by_coord(lon, lat, folder=f"./{city_pinyin}")
                if fn is not None:
                    print('\t', fn)
                    res.append( fn )
            except:
                error_lst.append([lon, lat])
                print("error, ", lon, ", ", lat)
        

if __name__ == '__main__':
    start()
    # get_pano_img_by_coord(116.403411,39.914001)
    # status, pano_respond, panos, nxt = query_pano( 12681303.05, 2582504.44, visualize=True, add_to_DB=False )
    # panos.iloc[1:-1].sample(1)
    