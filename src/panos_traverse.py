#%%
import os, sys
import math
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
import matplotlib.pyplot as plt
from road_network import OSM_road_network

from db.features_API import get_features
from db.db_process import load_from_DB
# from road_matching import traverse_panos_by_rid
from pano_img import get_pano_ids_by_rid, pano_dir, PANO_log
from utils.geo_plot_helper import map_visualize
from panos.panoAPI import get_panos

#%%

def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
    """Obtain the panos in road[rid] by distributed crawleers.

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
                continue

        # Fetch data asynchronously
        fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        params = {'pid': pid, 'heading': heading, 'path': fn}
        if os.path.exists(fn):
            # print('\texist: ',params)
            continue
        result = get_panos(params)
        res.append(result)
        pre_heading = heading
        
    return res, df_pids


def count_panos_num_by_area():
    """Count panos number in several area.

    Returns:
        [type]: [description]
    """
    areas = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs_with_Dapeng.geojson')
    res = {}
    for index, district in tqdm(areas.iterrows()):
        panos = get_features('point', geom = district.geometry)
        res[district.name_cn] = panos.shape[0]
        
    # df = gpd.sjoin(left_df=area, right_df=DB_panos, op='contains').groupby('name')[['DIR']].count()

    return res


def crawle_panos_in_district_area(district='南山区', key='name', fn='/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson'):
    area = gpd.read_file(fn)
    area.query( f"{key} =='{district}'", inplace=True )
    if area.shape[0] == 0:
        return None
    
    area = area.iloc[0].geometry
    features = get_features('line', geom = area)
    panos = get_features('point', geom = area)

    res    = []
    count = 0
    for rid in tqdm(features.RID.unique(), district):
        info, _ = traverse_panos_by_rid(rid, panos, log=PANO_log, all=True)
        res += info
        count += 1
        # if count > 500: break
    print(len(res))
    
    return res

def crawle_panos_in_city_level(fn='/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs_with_Dapeng.geojson'):
    areas = gpd.read_file(fn)
    for index, district in areas.iterrows():
        crawle_panos_in_district_area(district.name_cn, 'name_cn', fn)
    
    return True
    

#%%

if __name__ == '__main__':
    crawle_panos_in_city_level()

    # crawle_panos_in_district_area('盐田区')

    # count_panos_num_by_area()  
      
    {'福田区': 82795,
    '罗湖区': 46297,
    '南山区': 130939,
    '盐田区': 23611,
    '大鹏新区': 20622,
    '龙华区': 98013,
    '坪山区': 43334,
    '龙岗区': 142577,
    '宝安区': 163176,
    '光明新区': 44404}


