#%%
import os
import sys
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd 
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import point

from pano_base import pano_base_main
from pano_img import get_staticimage_batch
from pano_predict import pred_trajectory, PRED_MEMO, update_unpredict_panos
from panos_topo import combine_rids, get_panos_by_rids, get_trajectory_by_rid
from setting import CACHE_FOLDER, DIS_FACTOR, LXD_BBOX, SZU_BBOX, SZ_BBOX

from db.db_process import save_to_geojson
from utils.log_helper import LogHelper, logbook
from utils.geo_plot_helper import map_visualize
from utils.df_helper import load_df_memo, query_df
from db.db_process import save_to_geojson, save_to_db

sys.path.append("/home/pcl/traffic/MatchGPS2OSM/src")
from matching import st_matching
from setting import PANO_FOLFER
from DigraphOSM import Digraph_OSM, load_net_helper, gdf_to_geojson

HMM_FOLDER = "/home/pcl/traffic/MatchGPS2OSM/input"
logger = LogHelper(log_name='main.log').make_logger(level=logbook.INFO)

#%%

def filter_panos_by_road_type(road_type='primary', dis=35):
    
    df_edges = net.df_edges.query("road_type == @road_type")

    roads_levles = {}
    roads_levles[road_type] = gpd.clip(df_edges, futian_area)

    roads_levles[road_type].buffer(20*DIS_FACTOR).plot()

    road_mask = gpd.GeoDataFrame(roads_levles[road_type].buffer(dis*DIS_FACTOR), columns=['geometry'])
    road_mask.loc[:, 'road_type'] = road_type
    mask = road_mask.dissolve('road_type')

    tmp_indexes = gpd.clip(gdf_roads, mask.iloc[0].geometry, keep_geom_type=True).index
    roads_levles[road_type] = gdf_roads.loc[tmp_indexes]
    
    return roads_levles


def check_all_traj(traj_lst):
    err_lst = [] 

    for id in tqdm(range(0, 200)):
        try:
            rid = traj_lst[id]

            save_fn = os.path.join( "../debug/matching", f"{id:03d}_{rid}.jpg")
            rids     = get_trajectory_by_rid(rid, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
            traj     = get_panos_by_rids(rids, gdf_roads, gdf_panos, plot=False)
            pred_res = pred_trajectory(traj, df_pred_memo, aerial_view=False, combine_view=False, with_lanes=True)

            path     = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=False, save_fn=save_fn)
        except:
            err_lst.append(id)
    
    return


def check_single_traj(id, traj_lst):
    rid = traj_lst[id]

    save_fn = os.path.join( "../debug/matching", f"{id:03d}_{rid}.jpg")
    rids     = get_trajectory_by_rid(rid, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
    traj     = get_panos_by_rids(rids, gdf_roads, gdf_panos, plot=False)
    pred_res = pred_trajectory(traj, df_pred_memo, aerial_view=True, combine_view=True, with_lanes=True)

    pred_res['gdf']
    pred_res['combine_view']


    res      = st_matching(traj, net, name=str(id), plot=True, satellite=True, debug_in_levels=False, save_fn=None, top_k=5, georadius=50, logger=logger)
    res['rList']
    
    return pred_res, res


#%%
df_pred_memo = load_df_memo(PRED_MEMO)

# step 1: download OSM data
net = load_net_helper(bbox=SZ_BBOX, combine_link=True, reverse_edge=True, two_way_offeset=True, cache_folder='../../MatchGPS2OSM/cache')

# step 2: dowload pano topo
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
pano_base_res = pano_base_main(project_name='futian', geom=futian_area)
gdf_base  = pano_base_res['gdf_base']
gdf_roads = pano_base_res['gdf_roads']
gdf_panos = pano_base_res['gdf_panos']
map_visualize( pano_base_res['gdf_roads'], scale=.01 )

# step 3: download pano imgs
# pano_img_res = get_staticimage_batch(pano_base_res['gdf_panos'], 50, True)

# step 4: pano topo
traj_rid_lst, rid_2_start = combine_rids(gdf_base, gdf_roads, plot=False)
traj_lst = [ x for x in traj_rid_lst.keys()]


#%%
# step 5: predict trajectory
rid = 'c09f7e-7d97-05ec-65d2-3ce39b'
# rid = '0784de-4da2-3ff2-527a-f7967e'
# rid = 'c5905e-565e-1613-f0b2-077d8a'
rid = 'fd3456-ce06-10a9-6b28-02e115'
rids = get_trajectory_by_rid(rid, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
traj = get_panos_by_rids(rids, gdf_roads, gdf_panos, plot=False)

pred_res = pred_trajectory(traj, df_pred_memo, aerial_view=False, combine_view=False, with_lanes=True)
# pred_res.keys(); pred_res['gdf']; pred_res['aerial_view'] ;  pred_res['combine_view']

# step 6: HMM
path = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=False)

# step 7: data fusing
# get_and_filter_panos_by_osm_rid

# %%
if __name__ == '__main__':
    # DEBUG 
    save_to_geojson(traj, os.path.join(HMM_FOLDER, "traj_debug_case2.geojson"))

    #%%
    # rid = '550a27-40c5-f0d3-5717-a1907d' # 金田路福田地铁站附近
    # rid = 'edbf2d-e2f3-703f-4b9f-9d6819' # 深南大道-市民中心-东行掉头
    # rid = 'cb7422-27d2-c73b-b682-a12ebd' # 深南大道辅道-市民中心段-东行
    # rid = '24fd43-b288-813c-b717-c8f6f8' # 深南大道西行


    check_single_traj(4, traj_lst)


    """ debug in levels """
    res = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=True, save_fn=None, top_k=5)

    # traj = traj.sort_index(ascending=False).reset_index(drop=True)

    """ save to db """
    # save_to_db(gdf_roads, 'test_all')
    save_to_db(roads_levles[road_type], 'test_primary')
    save_to_db(mask, 'test_mask_primary')



# %%
net.df_edges.road_type.unique()
# %%
gdf_roads_levels = filter_panos_by_road_type('primary')
pids = np.unique(
            gdf_roads_levels['primary'].src.tolist() + \
            gdf_roads_levels['primary'].dst.tolist()
        ).tolist()

traj_rid_lst, rid_2_start = combine_rids(gdf_base.loc[pids], gdf_roads_levels['primary'], plot=True, logger=logger)


#%%

lst = {}

for key in tqdm(traj_rid_lst.keys()):
    lst[key] = {}
    lst[key]['rids'] = get_trajectory_by_rid(key, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
    lst[key]['rids_num'] = len(lst[key]['rids'])
    lst[key]['pids_df'] = get_panos_by_rids(lst[key]['rids'], gdf_roads, gdf_panos, plot=False)
    lst[key]['pids_num'] = lst[key]['pids_df'].shape[0]

df = pd.DataFrame(lst).T

df.rids = df.rids.astype(str)
df.drop_duplicates('rids', inplace=True)

df.rids = df.rids.apply(lambda x: eval(x))


# %%
df.sort_values('pids_num', ascending=False, inplace=True)

# %%
path = st_matching(df.iloc[3].pids_df, net, plot=True, satellite=True, debug_in_levels=False)


# %%
df.iloc[[8,9,10,11]]
# %%
rid_2_start['c8b1bb-e6c4-b2b3-b83f-9a5efd']
rid_2_start['24fd43-b288-813c-b717-c8f6f8']
rid_2_start['5da91b-c0c9-0588-150c-0e6bfb']


# %%


# %%

df.loc[:, 'status'] = df.apply(lambda x: x.rids.index(x.name), axis=1)
df.loc[:, 'status'].value_counts()

# df.iloc[8].rids.index(df.iloc[8].name)

# %%

df.loc[
    df[df.status==11].apply(lambda x: x['rids'][1], axis=1).values.tolist()
]
# %%

err_lst = [] # [125, 153, 174, 233]
for id in tqdm(range(125, 300)):
    try:
        path = st_matching(df.iloc[id].pids_df, net, plot=True, satellite=True, debug_in_levels=False, save_fn=f'../debug/matching/{id:03d}_{df.iloc[id].name}.jpg')
    except:
        err_lst.append(id)


# %%
