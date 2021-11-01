#%%
import os
import sys
import math
import heapq
import numpy as np
import pandas as pd 
from tqdm import tqdm
import geopandas as gpd
from scipy import stats
from shapely.geometry import point, LineString, box

from pano_base import pano_base_main
from pano_img import get_staticimage_batch
from panos_topo import combine_rids, Pano_UnionFind
from pano_predict import pred_trajectory, PRED_MEMO, update_unpredict_panos
from setting import CACHE_FOLDER, DIS_FACTOR, LXD_BBOX, SZU_BBOX, SZ_BBOX, FT_BBOX, link_type_no_dict

from utils.log_helper import LogHelper, logbook
from utils.geo_plot_helper import map_visualize
from utils.df_helper import load_df_memo, query_df, gdf_concat
from utils.interval_helper import merge_intervals_lst
from db.db_process import gdf_to_postgis, gdf_to_geojson

sys.path.append("/home/pcl/traffic/MatchGPS2OSM/src")
from matching import st_matching, cal_relative_offset
from DigraphOSM import Digraph_OSM, load_net_helper

HMM_FOLDER = "/home/pcl/traffic/MatchGPS2OSM/input"
logger = LogHelper(log_name='main.log').make_logger(level=logbook.INFO)

pd.set_option('display.max_rows', 50)

df_pred_memo = load_df_memo(PRED_MEMO)

"""特殊情况
    1. 单行线的影响：福中一路(益田路至民田路段)，由东向西单向通行
"""


#%%

def find_prev_edge(item, net, logger=None):
    if isinstance(item, int):
        item = net.df_edges.loc[item]
        
    res = gpd.GeoDataFrame()
    if item.order != 0:
        mid = net.df_edges.query(f"rid == {item.rid}")
        order_lst = sorted(mid.order.values.tolist())
        prev_order = order_lst[order_lst.index(item.order)-1]
        res = mid.query(f"order == {prev_order}")
    
    if item.order == 0 or len(res) == 0 :
        candidates = net.df_edges.query(f" e == {item.s} ")
        if candidates.shape[0] == 0:
            return None
        elif candidates.shape[0] == 1:
            res = candidates
        elif candidates.shape[0]> 1:
            candidates.query(f"road_type=='{item.road_type}'", inplace=True)
            if candidates.shape[0] == 1:
                res = candidates
            else:
                if logger is not None:
                    logger.error(f"check {item.eid}")
                raise 
                # return None
        
    return res.iloc[0].eid


def find_nxt_edge(item, net, logger=None):
    if isinstance(item, int):
        item = net.df_edges.loc[item]
        
    res = gpd.GeoDataFrame()
    mid = net.df_edges.query(f"rid == {item.rid}")
    order_lst = sorted(mid.order.values.tolist())
    if item.order != order_lst[-1]:
        nxt_order = order_lst[order_lst.index(item.order)+1]
        res = mid.query(f"order == {nxt_order}")
    
    if item.order == 0 or len(res) == 0 :
        candidates = net.df_edges.query(f" s == {item.e} ")
        if candidates.shape[0] == 0:
            return None
        elif candidates.shape[0] == 1:
            res = candidates
        elif candidates.shape[0]> 1:
            candidates.query(f"road_type=='{item.road_type}'", inplace=True)
            if candidates.shape[0] == 1:
                res = candidates
            else:
                if logger is not None:
                    logger.error(f"check {item.eid}")
                raise
                # FIXME
                # return None
        
    return res.iloc[0].eid


#%%
""" post mathcing """
def modify_laneNum_based_on_edge_type(item):
    """
    case:
        1. 终点是信号灯，若没有记录，考虑路段的长度，以及起点的分支情况，若有则减少
        2. 起点是信号灯，若没有记录，考虑上一路段的情况
        3. 标准路口范围内的link，全部设置为待定
    """ 
    
    
    pass


def _get_lane_set(df, df_pid_2_edge, prev_top_k=2, nxt_top_k=1):
    df = df.copy()
    df.loc[:, 'prev'] = df.apply(lambda x: find_prev_edge(x, net, logger), axis=1)
    df.loc[:, 'nxt'] = df.apply(lambda x: find_nxt_edge(x, net, logger), axis=1)
    df.loc[:, 'prev_pids'] = df.prev.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=False).head(prev_top_k).pid.values[::-1])
    df.loc[:, 'pids']      = df.eid.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).pid.values)
    df.loc[:, 'nxt_pids']  = df.nxt.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).head(nxt_top_k).pid.values)
    df.loc[:, 'combine_pids'] = df.apply(lambda x: list(x.prev_pids) + list(x.pids) + list(x.nxt_pids) , axis=1)
    df.loc[:, 'lane_set'] = df.combine_pids.apply(lambda x: [gdf_panos.loc[i].lane_num for i in x ])
    
    return df


def _indentify_lane_num_for_type_0(edge_miss, edges, df_pid_2_edge, prev_top_k=2, nxt_top_k=1):
    """Process the records without lane_num data

    Args:
        edge_miss ([type]): The records without lane_nums
        edges ([type]): The records with lane_nums
        top_k (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if edge_miss.shape[0] == 0:
        return edge_miss 
    
    df = edge_miss.copy()
    attrs = list(df)
    if 'lane_num' in df:
        df.drop(columns='lane_num', inplace=True)

    df = _get_lane_set(df, df_pid_2_edge, prev_top_k, nxt_top_k)
    df[['eid', 'prev', 'nxt', 'combine_pids', 'lane_set']]

    edge_to_pid_new = {}
    for index, item in df[['eid', 'combine_pids']].set_index('eid').iterrows():
        edge_to_pid_new[index] = item.combine_pids

    # FIXME None Data
    df_filtered = pids_filter(sort_pids_in_edge(edge_to_pid_new, df_pred_memo, plot=False))
    df_lanes = pd.DataFrame(df_filtered[~df_filtered.outlier].groupby('eid')['lane_num'].apply(list)).rename(columns={'lane_num':"lane_set"})
    df_lanes.loc[:,'lane_num'] = df_lanes.lane_set.apply(_get_lst_mode)

    df = df.reset_index(drop=True).merge(df_lanes, how='left', on='eid').set_index('eid')

    still_miss_con = df.lane_num.isna()
    df.loc[still_miss_con, 'lane_num'] = df[still_miss_con].prev.apply(lambda x: edges.loc[x].lane_num)

    return df['lane_num']


def _indentify_lane_num_for_type_3(edges_, idxs, df_pid_2_edge, net, lst_name='lane_set',  logger=None):
    def helper(item):
        if len(item[lst_name]) == 0:
            return None
        
        lane_mode = _get_lst_mode(item[lst_name])
        if lane_mode is None:
            prev_eid = find_prev_edge(item, net)
            if logger is not None:
                logger.info(f"edge: {item.eid}, lane_num {lane_mode} (type 3) refered to the previous edge {prev_eid}")
            lane_mode = edges_.loc[prev_eid].lane_num
                
            return lane_mode
    
        return lane_mode
    
    return _get_lane_set(edges_.loc[idxs], df_pid_2_edge, 2, 0).apply(lambda x: helper(x), axis=1)


def handle_missing_lane_record(edge_miss, edges, df_pid_2_edge, prev_top_k=2, nxt_top_k=1):
    """Process the records without lane_num data

    Args:
        edge_miss ([type]): The records without lane_nums
        edges ([type]): The records with lane_nums
        top_k (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    if edge_miss.shape[0] == 0:
        return edge_miss 
    
    df = edge_miss.copy()
    attrs = list(df)
    if 'lane_num' in df:
        df.drop(columns='lane_num', inplace=True)

    df = _get_lane_set(df, df_pid_2_edge, prev_top_k, nxt_top_k)
    df[['eid', 'prev', 'nxt', 'combine_pids', 'lane_set']]

    edge_to_pid_new = {}
    for index, item in df[['eid', 'combine_pids']].set_index('eid').iterrows():
        edge_to_pid_new[index] = item.combine_pids

    # FIXME None Data
    df_filtered = pids_filter(sort_pids_in_edge(edge_to_pid_new, df_pred_memo, plot=False))
    df_lanes = pd.DataFrame(df_filtered[~df_filtered.outlier].groupby('eid')['lane_num'].apply(list)).rename(columns={'lane_num':"lane_set"})
    df_lanes.loc[:,'lane_num'] = df_lanes.lane_set.apply(_get_lst_mode)

    df = df.reset_index(drop=True).merge(df_lanes, how='left', on='eid')

    still_miss_con = df.lane_num.isna()
    df.loc[still_miss_con, 'lane_num'] = df[still_miss_con].prev.apply(lambda x: edges.loc[x].lane_num)

    return df[attrs] if 'lane_num' in attrs else df[attrs+['lane_num']]


def get_laneNum(edges, db_name=None):
    df_pid_2_edge = pids_filter(
        sort_pids_in_edge(EDGE_TO_PID, df_pred_memo, plot=False)
    )

    #  df_pid_2_edge.query("eid==53560")
    df_lane_nums = pd.DataFrame(
        df_pid_2_edge[~df_pid_2_edge.outlier]\
            .groupby('eid')['lane_num']\
            .apply(list)\
            .apply(_get_lst_mode)
    )

    # filter the unpredicted edge
    idxs = list(np.setdiff1d( edges.index, df_lane_nums.index))
    edges.loc[idxs]
    df_pid_2_edge.query("eid in @idxs")
    # map_visualize(edges.loc[idxs])

    edges_ = edges.merge(df_lane_nums, on='eid', how='left')
    edges_.index = edges_.eid
    edge_miss = edges_[edges_.lane_num.isna()]

    # case 0: normal links
    idxs = edge_miss[edge_miss.edge_type == 0].index
    # if len(idxs) > 0:
    #     edges_.loc[idxs, 'lane_num'] =  _indentify_lane_num_for_type_0(edge_miss, edges_, df_pid_2_edge )

    # case 3: signal controled edge
    idxs = edge_miss[edge_miss.edge_type == 0].index
    if len(idxs) > 0:
        edges_.loc[idxs, 'lane_num'] = _indentify_lane_num_for_type_3(edges_, idxs, df_pid_2_edge, net=net)

    if db_name is not None:
        gdf_to_postgis(edges_, db_name)

    return df_lane_nums, edges_



#%%
if __name__ == '__main__':
    # # DEBUG 
    # save_to_geojson(traj, os.path.join(HMM_FOLDER, "traj_debug_case2.geojson"))

    # #%%
    # # rid = '550a27-40c5-f0d3-5717-a1907d' # 金田路福田地铁站附近
    # # rid = 'edbf2d-e2f3-703f-4b9f-9d6819' # 深南大道-市民中心-东行掉头
    # # rid = 'cb7422-27d2-c73b-b682-a12ebd' # 深南大道辅道-市民中心段-东行
    # # rid = '24fd43-b288-813c-b717-c8f6f8' # 深南大道西行


    # check_single_traj(4, traj_lst)


    """ debug in levels """
    # res = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=True, save_fn=None, top_k=5)

    # # traj = traj.sort_index(ascending=False).reset_index(drop=True)

    """ save to db """
    # # gdf_to_postgis(gdf_roads, 'test_all')
    # gdf_to_postgis(mask, 'test_mask_primary')
    
    
    """upload to db for debug"""
    # gdf_to_postgis( gpd.GeoDataFrame(df_trajs).drop(columns='pids_df'), 'test_topo_combine' )

    """for test"""
    # rid = 'e500bb-bfe5-17ae-b949-fc8d2a'
    # traj_uf.get_traj(rid, plot=True)

    """step 4: pano topo"""
    # uf, df_topo, df_topo_prev  = combine_rids(gdf_base, gdf_roads, gdf_panos, plot=False)
    # traj_lst = uf.trajs_to_gdf()


    """step 5: predict trajectory"""
    # rid = 'e500bb-bfe5-17ae-b949-fc8d2a'
    # rids = traj_lst.index

    # rid = rids[10]
    # traj = uf.get_panos(rid, plot=True)

    # pred_res = pred_trajectory(traj, df_pred_memo, aerial_view=False, combine_view=False, with_lanes=True)
    # pred_res.keys(); pred_res['gdf']; pred_res['aerial_view'] ;  pred_res['combine_view']

    """step 6: HMM"""
    # path = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=False)

    """step 7: data fusing"""
    # get_and_filter_panos_by_osm_rid

    pass


#%%
