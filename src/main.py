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
from shapely.geometry import point

from pano_base import pano_base_main
from pano_img import get_staticimage_batch
from pano_predict import pred_trajectory, PRED_MEMO, update_unpredict_panos
from panos_topo import combine_rids, get_panos_by_rids, get_trajectory_by_rid
from setting import CACHE_FOLDER, DIS_FACTOR, LXD_BBOX, SZU_BBOX, SZ_BBOX

from db.db_process import gdf_to_postgis, save_to_geojson
from utils.log_helper import LogHelper, logbook
from utils.geo_plot_helper import map_visualize
from utils.df_helper import load_df_memo, query_df

sys.path.append("/home/pcl/traffic/MatchGPS2OSM/src")
from matching import st_matching, cal_relative_offset
from DigraphOSM import Digraph_OSM, load_net_helper, gdf_to_geojson

HMM_FOLDER = "/home/pcl/traffic/MatchGPS2OSM/input"
logger = LogHelper(log_name='main.log').make_logger(level=logbook.INFO)

pd.set_option('display.max_rows', 20)


#%%

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


def get_pano_trajs(traj_rid_lst, rid_2_start, gdf_roads, gdf_panos):
    lst = {}
    for key in tqdm(traj_rid_lst.keys()):
        lst[key] = {}
        lst[key]['rids'] = get_trajectory_by_rid(key, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
        lst[key]['rids_num'] = len(lst[key]['rids'])
        lst[key]['pids_df'] = get_panos_by_rids(lst[key]['rids'], gdf_roads, gdf_panos, plot=False)
        lst[key]['pids_num'] = lst[key]['pids_df'].shape[0]
        # if lst[key]['pids_df'] is not None:
            # lst[key]['pred'] = pred_trajectory(lst[key]['pids_df'], df_pred_memo, aerial_view=False, combine_view=True, with_lanes=True)

    df = pd.DataFrame(lst).T
    df.rids = df.rids.astype(str)
    df.drop_duplicates('rids', inplace=True)
    df.rids = df.rids.apply(lambda x: eval(x))

    df.sort_values('pids_num', ascending=False, inplace=True)
    
    return df


def get_pano_topo_by_road_level(net, road_type='primary', dis=35, filter_sql=None, clip_geom=None):
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
    traj_rid_lst, rid_2_start = combine_rids(gdf_base.loc[pids], gdf_roads_levels[road_type], plot=True, logger=logger)

    df_trajs = get_pano_trajs(traj_rid_lst, rid_2_start, gdf_roads, gdf_panos)

    return df_trajs, traj_rid_lst, rid_2_start, df_edges


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


# %%
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



# %%
# net.df_edges.road_type.unique()

PANO_RID_VISITED = set()

PID_TO_EDGE = {}
EDGE_TO_PID = {}

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


def rid_heap_helper(item, net):
    """create the heap item for road traverse.

    Args:
        item ([type]): [description]
        net ([type]): [description]

    Returns:
        [type]: [description]
    """
    record = (
        gdf_panos.query(f"RID==@item.name").distance(net.df_edges.loc[item['eid']].geometry).mean() / DIS_FACTOR,
        item['eid'],
        item.name,
    )

    return record


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


def sort_pids_in_edge(edge_to_pid, plot=True):
    df_mapping = pd.DataFrame([edge_to_pid]).T.rename(columns={0: "pid"}).explode('pid')

    df_mapping.loc[:, 'offset'] = df_mapping.apply(
        lambda x: cal_relative_offset(gdf_panos.loc[x.pid].geometry, net.df_edges.loc[x.name].geom_origin)[0] / net.df_edges.loc[x.name].dist, 
        axis=1
    )

    df_mapping = df_mapping.reset_index().rename(columns={'index': 'eid'}).sort_values(['eid', 'offset'])

    if plot:
        map_visualize(
            gdf_panos.loc[ df_mapping.pid]
        )

    return df_mapping


def pids_filter(points, outlier_filter=True, mul_factor=2, verbose=True):
    def _panos_filter(panos, trim_nums=0):
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
        if panos.shape[0] == 2 and panos.lane_num.nunique() == 1:
            return panos

        # panos.reset_index(drop=True)
        median = int(np.median(panos.lane_num))
        remain_ponas_index = np.sort(panos.index)[trim_nums: -trim_nums] if trim_nums != 0 else np.sort(panos.index)

        tmp = panos[['offset','lane_num']]
        prev = panos.lane_num.shift(-1) == panos.lane_num
        nxt = panos.lane_num.shift(1) == panos.lane_num
        not_continuous = tmp[(prev|nxt) == False].offset.values.tolist()
        
        idxs = panos.query( f" offset not in {not_continuous} \
                        and index in @remain_ponas_index \
                        and abs(lane_num - @median) < 2", 
                        # inplace=True 
                    ).index
        
        panos.loc[idxs, 'outlier'] = False
        
        return panos
    
    # points = DB_panos.query( f"RID in {rids}" ).dropna()
    points.loc[:, 'outlier'] = True
    if outlier_filter and points.shape[0] != 0:
        if verbose: 
            origin_size = points.shape[0]
            
        _mean, _std = points.lane_num.mean(), points.lane_num.std()
        if not np.isnan(_mean) and not np.isnan(_std):
            iterverl = (_mean-mul_factor*_std, _mean+mul_factor*_std)
            idxs = points.query( f" {iterverl[0]} < lane_num < {iterverl[1]}").index
            points.loc[idxs, 'outlier'] = False

    points = points.groupby('eid').apply( lambda x: _panos_filter(x) )
    
    return points.reset_index()
    

def update_lanenum_of_edge():
    # TODO
    pass


def find_prev_edge(item, net=net, logger=logger):
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


def find_nxt_edge(item, net=net, logger=logger):
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



#%%
""" initialize """
df_trajs, traj_rid_lst, rid_2_start, edges = get_pano_topo_by_road_level(net, 'primary', filter_sql="name == '滨河大道辅路'", clip_geom=futian_area)



# %%
"""s1: select the top 20% trajectories for matching"""
ST_MATCHING_DICT = {} # {rid: mathching_path}

def mathicng_for_level(df_trajs, top_percentage=.1, coverage_thred=.6):
    for id in tqdm(range(math.ceil(df_trajs.shape[0]*top_percentage))):
        path = st_matching(df_trajs.iloc[id].pids_df, net, plot=False, satellite=False)
        ST_MATCHING_DICT[df_trajs.iloc[id].name] = path['path']
        map_pid_to_edge(df_trajs.iloc[id].pids_df, path['path'], PID_TO_EDGE, EDGE_TO_PID)
        
        for rid in df_trajs.iloc[id].rids:
            PANO_RID_VISITED.add(rid)

    MATCHING_MEMO = gpd.GeoDataFrame(pd.concat(ST_MATCHING_DICT.values()))
    # cal_coverage_helper(MATCHING_MEMO, format='dataframe')
    coverage_dict = cal_coverage_helper(MATCHING_MEMO, format='dict')


    """ deal with the unmatching edge """
    df_edges_unvisted = get_unvisited_edge(edges, MATCHING_MEMO)
    related_rids = get_unvisited_edge_related_rids(df_edges_unvisted, gdf_roads)

    lst = df_edges_unvisted.eid.unique()

    eid_visited = set()

    for idx in range(len(lst)):
        queue = related_rids.query(f'eid=={lst[idx]}').apply(lambda x: rid_heap_helper(x, net), axis=1).values.tolist()

        if len(queue) == 0:
            continue
        
        heapq.heapify(queue)
        plot_neighbor_rids_of_edge(lst[idx], df_edges_unvisted)

        tmp = None
        while queue:
            _, eid, rid = heapq.heappop(queue) 
            if eid in eid_visited:
                continue
            
            ori_coverage = coverage_dict.get(eid, None) 
            if ori_coverage is not None and ori_coverage['percentage'] > coverage_thred:
                continue
            
            print(f"{eid}: {ori_coverage['percentage'] if ori_coverage is not None else ''}, {rid}")

            save_fn = os.path.join( "../debug/matching", f"{eid}_{rid}.jpg")
            rids = get_trajectory_by_rid(rid, rid_2_start, traj_rid_lst, gdf_roads, plot=False)
            traj = get_panos_by_rids(rids, gdf_roads, gdf_panos, plot=False)
            # res  = st_matching(traj, net, name=str(id), plot=True, satellite=True, debug_in_levels=False, save_fn=save_fn, top_k=5, georadius=50, logger=logger)
            res  = st_matching(traj, net, name=str(id), plot=False, top_k=5, georadius=50, logger=logger)

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
            # TODO drop_duplicates
            
            ST_MATCHING_DICT[rid] = res['path']
            map_pid_to_edge(traj, res['path'], PID_TO_EDGE, EDGE_TO_PID)
            MATCHING_MEMO = pd.concat([MATCHING_MEMO, res['path']])

            if new_coverage > coverage_thred:
                eid_visited.add(eid)
                print(f'add {eid}')
                
                break
        else:
            print(f"can't meet the demand")
            

mathicng_for_level(df_trajs)

#%%

# EDGE_TO_PID[43721] # 6，7，5 没有众数

df_pid_2_edge = sort_pids_in_edge(EDGE_TO_PID, plot=False)
df_pid_2_edge = df_pid_2_edge.merge(df_pred_memo[['PID', 'lane_num']], left_on='pid', right_on='PID').drop(columns='PID')


df_pid_2_edge = pids_filter(df_pid_2_edge)
df_pid_2_edge.drop_duplicates(['eid', 'pid'], inplace=True)

df_pid_2_edge

df_lane_nums = pd.DataFrame(df_pid_2_edge.query('outlier==False').groupby('eid')['lane_num'].apply(list).apply(lambda x: stats.mode(x)[0][0]-1))

edges = edges.merge(df_lane_nums, on='eid', how='left')
edges[edges.lane_num.isna()]

gdf_to_postgis(
    net.df_edges.merge(df_lane_nums, on='eid'),
    'test_lane_res'
)

# gdf_to_postgis(
#     edges[edges.lane_num.isna()],
#     'test_lane_no_record'
# )

#%%
# TODO 上下游 - 连接

edge_miss = edges[edges.lane_num.isna()]

edge_miss

eid = 44687
EDGE_TO_PID[eid]

df_pid_2_edge.query(f"eid == {eid}")


#%%
item = edge_miss.iloc[4]

find_prev_edge(item, net)

edge_miss.loc[:, 'prev'] = edge_miss.apply(find_prev_edge, axis=1)
edge_miss.loc[:, 'nxt'] = edge_miss.apply(find_nxt_edge, axis=1)

edge_miss[['prev','eid', 'nxt']]

top_k = 1
edge_miss.loc[:, 'prev_pids'] = edge_miss.prev.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=False).head(top_k).pid.values)
edge_miss.loc[:, 'pids'] = edge_miss.eid.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).pid.values)
edge_miss.loc[:, 'nxt_pids'] = edge_miss.nxt.apply(lambda x: df_pid_2_edge.query(f"eid=={x}").sort_values('offset', ascending=True).head(top_k).pid.values)


#%%
edge_miss[['prev','eid', 'nxt', 'prev_pids', 'pids','nxt_pids']]


#%%
eid = 43733

EDGE_TO_PID[43733]


df_pid_2_edge.query(f"eid == {eid}")

df_lane_nums.loc[eid]


#%%




"""step 3: download pano imgs"""
# pano_img_res = get_staticimage_batch(pano_base_res['gdf_panos'], 50, True)

"""step 4: pano topo"""
traj_rid_lst, rid_2_start = combine_rids(gdf_base, gdf_roads, plot=False)
traj_lst = [ x for x in traj_rid_lst.keys()]


#%%
"""step 5: predict trajectory"""
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



#%%%



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



