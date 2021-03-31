import os, sys
import math
import shutil
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from road_network import OSM_road_network

from db.features import get_features
from db.db_process import load_from_DB
from road_matching import traverse_panos_by_rid
from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir
from utils.geo_plot_helper import map_visualize
from pano_img import PANO_log

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)


def get_panos_imgs_by_bbox():
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    res    = []
    
    bbox=[113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
    bbox = [113.92131,22.52442, 113.95630,22.56855] # 科技园片区
    # bbox=[113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
    bbox = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
    
    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
    area = area.query( "name =='龙华区'" )  
    
    features = get_features('line', geom = area.iloc[0].geometry)

    # features = get_features('line', bbox=bbox)
    for rid in tqdm(features.RID.unique()):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log, all=True)
        res += info
    len(res)
    
    # if not os.path.exists(folder): os.mkdir(folder)

    # for fn in res:
    #     shutil.copy( fn, folder )

    # print( 'total number of pano imgs: ', len(res))

    # cmd = os.popen( f" mv {folder} {dst} " ).read()

    return res

#%%

if __name__ == '__main__':
        
    get_panos_imgs_by_bbox()


