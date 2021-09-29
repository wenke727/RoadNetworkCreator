#%%
import os
import sys
import math
import heapq
import numpy as np
import pandas as pd 
from copy import copy
from tqdm import tqdm
import geopandas as gpd
from scipy import stats
from collections import deque
from shapely.geometry import point, LineString

from pano_base import pano_base_main
from pano_img import get_staticimage_batch
from panos_topo import combine_rids, Pano_UnionFind
from setting import CACHE_FOLDER, DIS_FACTOR, LXD_BBOX, SZU_BBOX, SZ_BBOX
from pano_predict import pred_trajectory, PRED_MEMO, update_unpredict_panos

from utils.log_helper import LogHelper, logbook
from utils.geo_plot_helper import map_visualize
from utils.df_helper import load_df_memo, query_df
from db.db_process import gdf_to_postgis, save_to_geojson

sys.path.append("/home/pcl/traffic/MatchGPS2OSM/src")
from matching import st_matching, cal_relative_offset
from DigraphOSM import Digraph_OSM, load_net_helper, gdf_to_geojson

HMM_FOLDER = "/home/pcl/traffic/MatchGPS2OSM/input"
logger = LogHelper(log_name='main.log').make_logger(level=logbook.INFO)

pd.set_option('display.max_rows', 50)


#%%

def gdf_concat(lst):
    return gpd.GeoDataFrame(pd.concat(lst))


def filter_panos_by_road_type(net, road_type='primary', dis=35, filter_sql=None, clip_geom=None):
    df_edges = net.df_edges.query("road_type == @road_type")
    if filter_sql is not None:
        df_edges.query(filter_sql, inplace=True)
    if clip_geom is not None:
        df_edges = gpd.clip(df_edges, futian_area)
        
    roads_levles = {}
    # TODO clip 和 sjoin 统一
    roads_levles[road_type] = gpd.clip(df_edges, futian_area)

    roads_levles[road_type].buffer(20*DIS_FACTOR).plot()

    road_mask = gpd.GeoDataFrame(roads_levles[road_type].buffer(dis*DIS_FACTOR), columns=['geometry'])
    road_mask.loc[:, 'road_type'] = road_type
    mask = road_mask.dissolve('road_type')

    tmp_indexes = gpd.clip(gdf_roads, mask.iloc[0].geometry, keep_geom_type=True).index
    roads_levles[road_type] = gdf_roads.loc[tmp_indexes]
    
    return roads_levles, df_edges


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


def get_pano_topo_by_road_level(net, gdf_base, gdf_panos, road_type='primary', dis=35, filter_sql=None, clip_geom=None):
    """get pano topo by level/sql/geom.

    Args:
        net ([type]): [description]
        road_type (str, optional): [description]. Defaults to 'primary'.
        dis (int, optional): [description]. Defaults to 35.
        filter_sql ([type], optional): [description]. Defaults to None.
        clip_geom ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    gdf_roads_levels, df_edges = filter_panos_by_road_type(net, road_type, filter_sql=filter_sql, clip_geom=clip_geom, dis=dis)
    pids = np.unique(
                gdf_roads_levels[road_type].src.tolist() + \
                gdf_roads_levels[road_type].dst.tolist()
            ).tolist()
    uf, df_topo, df_topo_prev = combine_rids(gdf_base.loc[pids], gdf_roads_levels[road_type], gdf_panos, plot=True, logger=logger)

    df_trajs = uf.trajs_to_gdf()

    return df_trajs, uf, df_edges


def merge_intervals(intervals):
    # intervals = [ [i[0], i[1]] for i in intervals ]
    intervals = sorted(intervals)
    result = []

    for interval in intervals:
        if len(result) == 0 or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            result[-1][1] = max(result[-1][1], interval[1])
            
    return result


#%%

""" 匹配中操作 """
def breakpoint_to_interval(item):
    """transfer the the first/last step breakpoint to coverage intervals.

    Args:
        item ([type]): [description]

    Returns:
        [type]: [description]
    """
    if item.step == 1:
        return [0, 1]
    if item.step == 0:
        return [item.breakpoint, 1]
    if item.step == -1:
        return [0, item.breakpoint]


def plot_neighbor_rids_of_edge(eid, df_edges_unvisted, dis_buffer=25):
    """Plot the neighbor rids of the eid.

    Args:
        eid ([type]): [description]
        df_edges_unvisted ([type]): [description]
    """
    # 测试 sjoin的功能, 发现主要引发bug的地方是 道路被切了
   
    road_mask = gpd.GeoDataFrame({'eid': eid, 'geometry': df_edges_unvisted.query(f'eid=={eid}').buffer(dis_buffer*DIS_FACTOR)})
    tmp = gpd.sjoin(gdf_roads, road_mask, op='intersects')
    tmp.reset_index(inplace=True)
    
    fig, ax = map_visualize(tmp)
    tmp.plot(column='ID', ax=ax, legend=True)
    df_edges_unvisted.query(f'eid=={eid}').plot(ax=ax, color='blue', linestyle='--')
    
    return


def cal_coverage_helper(df, coverage_thred=.6, format='dataframe'):
    """transfer the matching dataframe  

    Args:
        df ([type]): [description]
        coverage_thred (float, optional): [description]. Defaults to .6.
        format (str, optional): [description]. Defaults to 'dataframe'.

    Returns:
        [type]: [description]
    """
    if df is None:
        return None
    assert format in ['dataframe', 'dict'], "Format parameter expect ['dataframe', 'dict']"
    
    df_ = df.copy()
    df_.loc[:, 'intervals'] = df_.apply(breakpoint_to_interval, axis=1)
    # TODO 针对起点和终点都是同一个路段的情况，区间会有错误，如：49691
    if df_.shape[0] == 2 and df_.eid.nunique() == 1:
        df_.iloc[:, ]['intervals'] = [[df_.iloc[0].intervals[0], df_.iloc[1].intervals[1]], [0, 0]]
    
    df_ = pd.DataFrame(
            df_[['eid', 'intervals']].groupby(['eid']).intervals.apply(lambda x: sorted(list(x)))
        ).sort_values('intervals')
    df_.loc[:, 'intervals_merged'] = df_.intervals.apply(merge_intervals)
    df_.loc[:, 'percentage'] = df_.intervals_merged.apply(lambda x: sum( [ i[1]-i[0] for i in x ]))
    df_.loc[:, 'visited'] = df_.percentage.apply(lambda x: True if x > coverage_thred else False)
    df_.sort_values('percentage', inplace=True)
    
    if format == 'dict':
        return df_.to_dict(orient='index')
    
    return df_


def get_unvisited_edge(edges, MATCHING_MEMO, plot=True):
    mathing_ = cal_coverage_helper(MATCHING_MEMO, format='dataframe')
    edges_ = edges.merge(mathing_, on=['eid'], how='left')
    edges_.visited = edges_.visited.fillna(False)
    
    if plot:
        map_visualize(edges_[~edges_.visited])
        
    return edges_[~edges_.visited]


def get_unvisited_edge_related_rids(df_edges_unvisted, gdf_roads, dis_buffer=25):
    # TODO add cos similarity
    road_mask = gpd.GeoDataFrame({
        'eid': df_edges_unvisted['eid'].values, 
        'geometry':df_edges_unvisted.buffer(dis_buffer*DIS_FACTOR)}
    )
    related_rids = gpd.sjoin(gdf_roads, road_mask, op='intersects', )

    return related_rids


""" 匹配后操作 """
def map_pid_to_edge(pids, route, pid_to_edge, edge_to_pid ):
    """map pid to edge based on the result of map matching

    Args:
        pids ([type]): [description]
        route ([type]): [description]
    """

    pids.loc[:, 'closest_eid'] = pids.apply(lambda x: route.loc[route.distance(x.geometry).idxmin()].eid , axis=1)

    for i, item in pids.iterrows():
        edge_to_pid[item['closest_eid']] = edge_to_pid.get(item['closest_eid'], [])
        if item['PID'] not in edge_to_pid[item['closest_eid']]:
            edge_to_pid[item['closest_eid']].append(item['PID'])
        
        pid_to_edge[item['PID']] = pid_to_edge.get(item['PID'], set())
        if item['closest_eid'] not in pid_to_edge[item['PID']]:
            pid_to_edge[item['PID']].add(item['closest_eid'])

    return


def sort_pids_in_edge(edge_to_pid, df_pred_memo=None, plot=True):
    df_mapping = pd.DataFrame([edge_to_pid]).T.rename(columns={0: "pid"}).explode('pid')

    df_mapping.loc[:, 'offset'] = df_mapping.apply(
        lambda x: cal_relative_offset(gdf_panos.loc[x.pid].geometry, net.df_edges.loc[x.name].geom_origin)[0] / net.df_edges.loc[x.name].dist, 
        axis=1
    )

    # df_mapping.drop_duplicates(['eid', 'pid'], inplace=True)
    df_mapping = df_mapping.reset_index().rename(columns={'index': 'eid'}).sort_values(['eid', 'offset'])
    
    if df_pred_memo is not None:
        df_mapping = df_mapping.merge(df_pred_memo[['PID', 'lane_num']], left_on='pid', right_on='PID').drop(columns='PID')

    if plot:
        map_visualize( gdf_panos.loc[ df_mapping.pid] )

    return df_mapping


def pids_filter(points, outlier_filter=True, mul_factor=2):
    def _panos_filter(points, trim_nums=0):
        """Filter panos by:
        1. trim port
        1. lane_detection continues
        1. abs(lane_num - @median) < 2

        Args:
            panos (df): The origin df.
            trim_nums (int, optional): the trim length of the begining adn end points. Defaults to 1.

        Returns:
            [pd.df]: the filtered panos
        """
        if points.shape[0] == 2 and points.lane_num.nunique() == 1:
            points.loc[:, 'outlier'] = False
            return points

        # panos.reset_index(drop=True)
        median = int(np.median(points.lane_num))
        remain_ponas_index = np.sort(points.index)[trim_nums: -trim_nums] if trim_nums != 0 else np.sort(points.index)

        tmp = points[['offset','lane_num']]
        prev = points.lane_num.shift(-1) == points.lane_num
        nxt = points.lane_num.shift(1) == points.lane_num
        not_continuous = tmp[(prev|nxt) == False].offset.values.tolist()
        
        idxs = points.query( f"not (offset not in {not_continuous} \
                        and index in @remain_ponas_index \
                        and abs(lane_num - @median) < 2)", 
                        # inplace=True 
                    ).index.tolist()
        logger.debug(f"Outlier index: {idxs}")
        points.loc[idxs, 'outlier'] = True
        
        return points
    
    def _panos_outlier(points):
        if outlier_filter and points.shape[0] != 0 and points.lane_num.nunique() > 1:
            # std 针对单一的数值的计算值为nan
            _mean, _std = points.lane_num.mean(), points.lane_num.std()
            if not np.isnan(_mean) and not np.isnan(_std):
                iterverl = (_mean-mul_factor*_std, _mean+mul_factor*_std)
                idxs = points.query( f"not ({iterverl[0]} < lane_num < {iterverl[1]})").index
                points.loc[idxs, 'outlier'] = True

        return points

    points.loc[:, 'outlier'] = False
    # logger.info(f"\n{points}")
    points = points.groupby('eid').apply( _panos_outlier )
    # logger.info(f"\n{points}")
    points = points.groupby('eid').apply( _panos_filter )
    # logger.info(f"\n{points}")
    
    return points


def mathicng_for_level(df_trajs, traj_uf, top_percentage=.1, coverage_thred=.6, debug=True):
    for id in tqdm(range(math.ceil(df_trajs.shape[0]*top_percentage))):
        if debug:
            save_fn = os.path.join( "../debug/matching", f"{id}_{df_trajs.iloc[id].name}.jpg")
            path = st_matching(df_trajs.iloc[id].pids_df, net, plot=True, satellite=True, save_fn=save_fn)
        else:
            path = st_matching(df_trajs.iloc[id].pids_df, net, plot=False, satellite=False)
        ST_MATCHING_DICT[df_trajs.iloc[id].name] = path['path']
        map_pid_to_edge(df_trajs.iloc[id].pids_df, path['path'], PID_TO_EDGE, EDGE_TO_PID)
        
        for rid in df_trajs.iloc[id].rids:
            PANO_RID_VISITED.add(rid)

    MATCHING_MEMO = gpd.GeoDataFrame(pd.concat(ST_MATCHING_DICT.values()))
    coverage_dict = cal_coverage_helper(MATCHING_MEMO, format='dict')

    # deal with the unmatching edge
    df_edges_unvisted = get_unvisited_edge(edges, MATCHING_MEMO)
    related_rids = get_unvisited_edge_related_rids(df_edges_unvisted, gdf_roads)
    lst = df_edges_unvisted.eid.unique()
    eid_visited = set()

    for idx in range(len(lst)):
        queue = related_rids.query(f'eid=={lst[idx]}').apply(lambda x: rid_heap_helper(x, net), axis=1).values.tolist()
        if len(queue) == 0:
            continue
        
        heapq.heapify(queue)
        # plot_neighbor_rids_of_edge(lst[idx], df_edges_unvisted)

        while queue:
            _, eid, rid = heapq.heappop(queue) 
            if eid in eid_visited:
                continue
            
            ori_coverage = coverage_dict.get(eid, None) 
            if ori_coverage is not None and ori_coverage['percentage'] > coverage_thred:
                continue
            
            print(f"{eid}: {ori_coverage['percentage'] if ori_coverage is not None else ''}, {rid}")

            save_fn = os.path.join( "../debug/matching", f"{eid}_{rid}.jpg")
            
            traj = traj_uf.get_panos(rid, False)
            res  = st_matching(traj, net, name=str(id), plot=True, satellite=True, debug_in_levels=False, save_fn=save_fn, top_k=5, georadius=50, logger=logger)
            # res  = st_matching(traj, net, name=str(id), plot=False, top_k=5, georadius=50, logger=logger)

            edge_related_new = net.df_edges.loc[[eid]][['eid']].merge(res['path'], on=['eid'])
            edge_related_old = net.df_edges.loc[[eid]][['eid']].merge(MATCHING_MEMO, on=['eid'])
            edge_related = pd.concat([edge_related_new, edge_related_old])
            if edge_related.shape[0] == 0:
                continue

            coverage_dict[eid] = cal_coverage_helper(edge_related, format='dict')[eid]
            new_coverage = coverage_dict[eid]['percentage']

            if ori_coverage is not None and new_coverage == ori_coverage['percentage']:
                continue
            
            print(f'\t new_coverage: {new_coverage}')
            
            ST_MATCHING_DICT[rid] = res['path']
            map_pid_to_edge(traj, res['path'], PID_TO_EDGE, EDGE_TO_PID)
            MATCHING_MEMO = pd.concat([MATCHING_MEMO, res['path']])

            if new_coverage > coverage_thred:
                eid_visited.add(eid)
                print(f'add {eid}')
                
                break
        else:
            print(f"can't meet the demand")
            

def find_prev_edge(item, net, logger=None):
    res = gpd.GeoDataFrame()
    if item.order != 0:
        res = net.df_edges.query(f"rid=={item.rid} and order == {item.order-1}")
    
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
                return None
        
    return res.iloc[0].eid


def find_nxt_edge(item, net, logger=None):
    res = gpd.GeoDataFrame()
    res = net.df_edges.query(f"rid=={item.rid} and order == {item.order+1}")
    
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
                return None
        
    return res.iloc[0].eid


def handle_missing_lane_record(edge_miss, edges, top_k=1):
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

    df.loc[:, 'prev'] = df.apply(lambda x: find_prev_edge(x, net), axis=1)
    df.loc[:, 'nxt'] = df.apply(lambda x: find_nxt_edge(x, net), axis=1)
    df.loc[:, 'prev_pids'] = df.prev.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=False).head(top_k).pid.values)
    df.loc[:, 'pids']      = df.eid.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).pid.values)
    df.loc[:, 'nxt_pids']  = df.nxt.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).head(top_k).pid.values)
    df.loc[:, 'combine_pids'] = df.apply(lambda x: list(x.prev_pids) + list(x.pids) + list(x.nxt_pids) , axis=1)
    # df[['prev','eid', 'nxt', 'prev_pids', 'pids','nxt_pids', 'combine_pids']]

    edge_to_pid_new = {}
    for index, item in df[['eid', 'combine_pids']].set_index('eid').iterrows():
        edge_to_pid_new[index] = item.combine_pids

    # FIXME None Data
    df_filtered = pids_filter(sort_pids_in_edge(edge_to_pid_new, df_pred_memo, plot=False))
    df_lanes = pd.DataFrame(df_filtered[~df_filtered.outlier].groupby('eid')['lane_num'].apply(list)).rename(columns={'lane_num':"lane_set"})
    # FIXME: stats.mode([5, 4, 4, 5]) -> 4
    df_lanes.loc[:,'lane_num'] = df_lanes.lane_set.apply(lambda x: stats.mode(x)[0][0]-1)

    df = df.merge(df_lanes, how='left', on='eid')
    still_miss_mask = df.lane_num.isna()
    df.loc[still_miss_mask, 'lane_num'] = df[still_miss_mask].prev.apply(lambda x: edges.query(f"eid=={x}").iloc[0].lane_num)

    return df[attrs] if 'lane_num' in attrs else df[attrs+['lane_num']]


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


    # """ debug in levels """
    # res = st_matching(traj, net, plot=True, satellite=True, debug_in_levels=True, save_fn=None, top_k=5)

    # # traj = traj.sort_index(ascending=False).reset_index(drop=True)

    # """ save to db """
    # # gdf_to_postgis(gdf_roads, 'test_all')
    # gdf_to_postgis(roads_levles[road_type], 'test_primary')
    # gdf_to_postgis(mask, 'test_mask_primary')
    pass


df_pred_memo = load_df_memo(PRED_MEMO)

"""step 1: download OSM data"""
net = load_net_helper(bbox=SZ_BBOX, combine_link=True, reverse_edge=True, two_way_offeset=True, cache_folder='../../MatchGPS2OSM/cache')

"""step 2: dowload pano topo"""
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
pano_base_res = pano_base_main(project_name='futian', geom=futian_area)
gdf_base  = pano_base_res['gdf_base']
gdf_roads = pano_base_res['gdf_roads']
gdf_panos = pano_base_res['gdf_panos']
map_visualize( pano_base_res['gdf_roads'], scale=.01 )

#%%
""" initialize """
# road_name = '深南大道' 
# road_name = '红荔路' 
# road_name = '滨河大道' 
road_name = '滨河大道辅路' 
road_name = '益田路' 
road_name = '福华路'

# roadtype = 'trunk', primary, secondary

df_trajs, traj_uf, edges = get_pano_topo_by_road_level(net, gdf_base, gdf_panos, 'secondary', filter_sql=f"name == '{road_name}'", clip_geom=futian_area)

# gdf_to_postgis( gpd.GeoDataFrame(df_trajs).drop(columns='pids_df'), 'test_topo_combine' )
# rid = 'e500bb-bfe5-17ae-b949-fc8d2a'
# traj_uf.get_traj(rid, plot=True)

#%%
# ! new traverse road framework
coverage_dict = {}
eid_visited = set()
MATCHING_MEMO = gpd.GeoDataFrame(columns=['eid'])
ST_MATCHING_DICT = {}
PID_TO_EDGE = {}
EDGE_TO_PID = {}



#%%


def get_laneNum(edges, plot=True, db_name=None):
    df_pid_2_edge = sort_pids_in_edge(EDGE_TO_PID, df_pred_memo, plot=False)
    df_pid_2_edge_filtered = pids_filter(df_pid_2_edge)
    df_lane_nums = pd.DataFrame(df_pid_2_edge_filtered[~df_pid_2_edge_filtered.outlier].groupby('eid')['lane_num'].apply(list).apply(lambda x: stats.mode(x)[0][0]-1))

    edges_ = edges.merge(df_lane_nums, on='eid', how='left')

    # TODO 上下游 - 连接
    edge_miss = edges_[edges_.lane_num.isna()]
    edge_miss_handled = handle_missing_lane_record(edge_miss, edges_ )

    df_lane_nums = gdf_concat([edges_[~edges_.lane_num.isna()], edge_miss_handled] )[['eid', 'lane_num']]
    edges_final = net.df_edges.merge(df_lane_nums, on='eid')

    if plot:
        fig, ax = map_visualize(edges_final, scale=.05)
        edges_final.plot(ax=ax, column='lane_num', legend=True, categorical=True)
    
    if db_name is not None:
        gdf_to_postgis( net.df_edges.merge(df_lane_nums, on='eid'), f'{db_name}')
        # gdf_to_postgis( edges_[edges_.lane_num.isna()], 'test_lane_no_record')
    
    return df_lane_nums


def matching_edge_helper(queue, coverage_thred=.7, logger=None):
    global MATCHING_MEMO
    visited_rids = set()
    while queue:
        _, eid, rid = heapq.heappop(queue) 
        print(eid, rid)
        
        if eid in eid_visited:
            break
        if rid in visited_rids:
            continue
        
        ori_coverage = coverage_dict.get(eid, None) 
        if ori_coverage is not None and ori_coverage['percentage'] > coverage_thred:
            print(f'{eid} had been matched')
            break

        rids = traj_uf.get_traj(rid)
        traj = traj_uf.get_panos(rid, False)
        if traj is None:
            if logger is not None:
                logger.warning(f"{eid}, {rid} has no matching trajectory")
            continue
        if logger is not None:
            logger.info(f"{eid}, {rid}, {ori_coverage['percentage'] if ori_coverage is not None else ''},")
            
        matching_res = st_matching(
            traj, 
            net, 
            name=str(id), 
            plot=True, 
            satellite=True, 
            save_fn=os.path.join( "../debug/matching", f"{eid}_{rid}.jpg"), 
            top_k=5, 
            logger=None
        )

        edge_related_new = net.df_edges[['eid']].merge(matching_res['path'], on=['eid'])
        if edge_related_new.query(f"eid=={eid}").shape[0] == 0:
            continue
        edge_related_old = net.df_edges[['eid']].merge(MATCHING_MEMO, on=['eid'])
        edge_related = pd.concat([edge_related_new, edge_related_old])

        new_cover_dict = cal_coverage_helper(edge_related, format='dict')
        for key, val in new_cover_dict.items():
            if val['percentage'] == 1:
                eid_visited.add(key)
            coverage_dict[key] = val
        new_coverage = coverage_dict[eid]['percentage']

        if ori_coverage is not None and new_coverage == ori_coverage['percentage']:
            continue
        
        print(f'\t new_coverage: {new_coverage}')
        ST_MATCHING_DICT[rid] = matching_res['path']
        map_pid_to_edge(traj, matching_res['path'], PID_TO_EDGE, EDGE_TO_PID)
        MATCHING_MEMO = pd.concat([MATCHING_MEMO, matching_res['path']])

        if new_coverage > coverage_thred:
            eid_visited.add(eid)
            for rid in rids:
                visited_rids.add(rid)
            print(f'add {eid}')
            
            break


def traverse_edges(edges, pano_roads, coverage_thred=.7, plot_candidates_rids=False, logger=None):
    def _rid_heap_helper(item, net):
        """create the heap item for road traverse.

        Args:
            item ([type]): [description]
            net ([type]): [description]

        Returns:
            [type]: [description]
        """
        record = (
            # TODO add similarity
            gdf_panos.query(f"RID==@item.name").distance(net.df_edges.loc[item['eid']].geometry).mean() / DIS_FACTOR,
            item['eid'],
            item.name,
        )

        return record

    eids_lst = edges.eid.unique().tolist()
    related_rids = get_unvisited_edge_related_rids(edges, pano_roads)

    for idx in range(len(eids_lst)):
        queue = related_rids.query(f'eid=={eids_lst[idx]}').\
                             apply(lambda x: _rid_heap_helper(x, net), axis=1).values.tolist()

        if len(queue) == 0:
            continue

        heapq.heapify(queue)
        if plot_candidates_rids:
            plot_neighbor_rids_of_edge(eids_lst[idx], edges)

        if logger is not None:
            logger.debug(queue)

        matching_edge_helper(queue, coverage_thred=coverage_thred, logger=logger)


traverse_edges(edges, gdf_roads, logger=logger)

#%%
get_laneNum(edges, db_name=f'test_lane_res{road_name}')



#%%

"""step 3: download pano imgs"""
# pano_img_res = get_staticimage_batch(pano_base_res['gdf_panos'], 50, True)

"""step 4: pano topo"""
traj_rid_lst, rid_2_start = combine_rids(gdf_base, gdf_roads, gdf_panos, plot=False)
traj_lst = [ x for x in traj_rid_lst.keys()]


#%%
"""step 5: predict trajectory"""
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
def pano_topo_mathcing_debug():
    debug_folder = '../debug/滨河大道/'

    df_trajs.query('rids_num >= 3 or pids_num >= 5 ', inplace=True)
    err_lst = [] # [44, 88]
    for id in tqdm(range(df_trajs.shape[0])):
        try:
            path = st_matching(df_trajs.iloc[id].pids_df, net, plot=True, satellite=True, debug_in_levels=False, 
                            traj_thres=5, save_fn=f'../debug/滨河大道_0/{id:03d}_{df.iloc[id].name}.jpg')
        except:
            err_lst.append(id)


    debug_folder = '../debug/滨河大道/'
    df_trajs.query('rids_num < 3 and pids_num < 5 ', inplace=True)
    err_lst = [] # [44, 88]
    for id in tqdm(range(df_trajs.shape[0])):
        try:
            path = st_matching(df_trajs.iloc[id].pids_df, net, plot=True, satellite=True, debug_in_levels=False, 
                            traj_thres=5, save_fn=f'../debug/滨河大道_1/{id:03d}_{df.iloc[id].name}.jpg')
        except:
            err_lst.append(id)


    err_img_lst = [] # [43, 85]
    debug_folder = '../debug/滨河大道/pred'

    for id in tqdm(range(df_trajs.shape[0])):
        try:
            traj = df_trajs.iloc[id].pids_df
            pred_res = pred_trajectory(traj, df_pred_memo, aerial_view=False, combine_view=True, with_lanes=True)
            # pred_res.keys(); pred_res['gdf']; pred_res['aerial_view'] ;  
            pred_res['combine_view'].save(os.path.join(debug_folder, f'{id:03d}_{df.iloc[id].name}.jpg') )
        except:
            err_img_lst.append(id)
            
    slight_rid = np.concatenate(df_trajs.query('rids_num <= 2 or pids_num <= 5 ').rids.values )
    gdf_roads.loc[slight_rid].plot()



