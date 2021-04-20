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
# from road_matching import traverse_panos_by_rid
from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir
from utils.geo_plot_helper import map_visualize
from pano_img import PANO_log
from panos.panoAPI import get_panos

# DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)



def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
    """obtain the panos in road[rid] 

    Args:
        rid (str): the id of road segements

    Returns:
        [type]: [description]
    """
    df_pids = get_pano_ids_by_rid(rid, DB_panos)
    
    pano_lst = df_pids[['Order','PID', 'DIR']].values
    length = len(pano_lst)
    res, pre_heading = [], 0
    
    for id, (order, pid, heading) in enumerate(pano_lst):
        if heading == 0 and id != 0:   # direction, inertial navigation
            heading = pre_heading
        
        if not all:
            if length > 3 and order % 3 != 1:
                # print(order)
                continue

        fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        if not os.path.exists(fn):
            params = {'pid': pid, 'heading': heading, 'path': fn}
            print('\t',params)
            result = get_panos(params)
            res.append(result)
            pre_heading = heading
        
    return res, df_pids


def count_panos_num_by_area():
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs_with_Dapeng.geojson')
    df = gpd.sjoin(left_df=area, right_df=DB_panos, op='contains').groupby('name')[['DIR']].count()
    
    return df


def get_panos_imgs_by_bbox():
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    
    # bbox=[113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
    # bbox = [113.92131,22.52442, 113.95630,22.56855] # 科技园片区
    # bbox=[113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
    bbox = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
    
    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
    area = area.query( "name =='龙岗区'" ).iloc[0].geometry
    # area = area.query( "name ==''" )  
    
    features = get_features('line', geom = area)
    panos = get_features('point', geom = area)

    # features = get_features('line', bbox=bbox)
    res    = []
    count = 0
    for rid in features.RID.unique():
        info, _ = traverse_panos_by_rid(rid, panos, log=PANO_log, all=True)
        res += info
        count += 1
        # if count > 500: break
    len(res)
    

    return res

#%%

if __name__ == '__main__':
    # get_panos_imgs_by_bbox()

    count_panos_num_by_area()    
    


