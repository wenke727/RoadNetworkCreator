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

""" filter panos """

def identify_edge_type(id, net, dis_thres=30):
    """通过规则识别路段的类型：交叉口等候区域，路口连接线

        # 普通路段
        for id in [20117,20122,20124,20128,20267,20273,20277,20279,20283,20290,]:
            link = net.df_edges.loc[id]
            print(net.a_star(link.e, link.s))

        # 判断是否为路口连接线, 常规路口
        for id in [20118,20269,20274,20284,20291,]:
            link = net.df_edges.loc[id]
            print( net.a_star(link.e, link.s) )

        # 掉头连接线
        link = net.df_edges.loc[2727]
        print( net.a_star(link.e, link.s) )

    Args:
        id ([type]): [description]
        net ([type]): [description]

    Returns:
        [type]: [description]
    """

    type_dict = {
        0: 'normal',
        1: 'U_turn_link',
        2: 'intersection_link',
        3: 'signal_control',
        4: 'edge_start_with_signal'
    }
    edge = net.df_edges.loc[id]
    if type(net.node[edge.e]['traffic_signals'])==str:
        return 3
    
    if type(net.node[edge.s]['traffic_signals'])==str:
        return 4

    path = net.a_star(edge.e, edge.s)
    if path is None or path['path'] is None:
        return 0
    
    if len(path['path']) == 2 and edge.dist < dis_thres and path['cost'] < dis_thres*2:
        return 1
    
    if len(path['path']) == 4 and edge.dist < dis_thres and path['cost'] < dis_thres*4:
        return 2
    
    return 0


def filter_panos_by_road(net, road_name=None, road_type='primary', dis=35, filter_sql=None, clip_geom=None):
    df_edges = net.df_edges.copy()
    if road_name is not None:
        df_edges = df_edges.query("name == @road_name")
    if road_name is None and road_type is not None:
        df_edges = df_edges.query("road_type == @road_type")
    if filter_sql is not None:
        df_edges.query(filter_sql, inplace=True)
    if clip_geom is not None:
        df_edges = df_edges.loc[ gpd.clip(df_edges, clip_geom).index ]
        
    road_mask = gpd.GeoDataFrame(df_edges.buffer(dis*DIS_FACTOR), columns=['geometry'])
    road_mask.loc[:, 'att'] = 1
    mask = road_mask.dissolve('att')

    idxs = gpd.clip(gdf_roads, mask.iloc[0].geometry, keep_geom_type=True).index
    
    return gdf_roads.loc[idxs], df_edges


def get_pano_topo_by_road(net, gdf_base, gdf_panos, road_name=None, road_type=None, dis=35, filter_sql=None, clip_geom=None):
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
    roads_levels, df_edges = filter_panos_by_road(net, road_name, road_type, filter_sql=filter_sql, clip_geom=clip_geom, dis=dis)
    df_edges.loc[:, 'edge_type'] = df_edges.eid.apply(lambda x: identify_edge_type(x, net))

    pids = np.unique(roads_levels.src.tolist() + roads_levels.dst.tolist()).tolist()
    uf, df_topo, df_topo_prev = combine_rids(gdf_base.loc[pids], roads_levels, gdf_panos, plot=True, logger=logger)

    df_trajs = uf.trajs_to_gdf()

    return df_trajs, uf, df_edges


#%%
""" 辅助函数 """
def plot_neighbor_rids_of_edge(eid, df_edges_unvisted, dis_buffer=25):
    """Plot the neighbor rids of the eid.

    Args:
        eid ([type]): [description]
        df_edges_unvisted ([type]): [description]
    """
    road_mask = gpd.GeoDataFrame({'eid': eid, 'geometry': df_edges_unvisted.query(f'eid=={eid}').buffer(dis_buffer*DIS_FACTOR)})
    tmp = gpd.sjoin(gdf_roads, road_mask, op='intersects')
    tmp.reset_index(inplace=True)
    
    fig, ax = map_visualize(tmp)
    tmp.plot(column='ID', ax=ax, legend=True)
    df_edges_unvisted.query(f'eid=={eid}').plot(ax=ax, color='blue', linestyle='--')
    
    return


""" 匹配中操作 """
def _breakpoint_to_interval(item):
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
    df_.loc[:, 'intervals'] = df_.apply(_breakpoint_to_interval, axis=1)
    
    if df_.shape[0] == 2 and df_.eid.nunique() == 1:
        logger.debug(f"检查针对起点和终点都是同一个路段的情况，区间是否会有错误，如：49691, 79749\n{df_}")
        df_.loc[:, 'intervals'] = pd.Series([[df_.iloc[0].intervals[0], df_.iloc[1].intervals[1]], [0, 0]])
    
    df_ = pd.DataFrame(
            df_[['eid', 'intervals']].groupby(['eid']).intervals.apply(lambda x: sorted(list(x)))
        ).sort_values('intervals')
    df_.loc[:, 'intervals_merged'] = df_.intervals.apply(merge_intervals_lst)
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


""" 匹配后操作 """
def map_pid_to_edge(pids, route, pid_to_edge, edge_to_pid ):
    """map pid to edge based on the result of map matching

    Args:
        pids ([type]): [description]
        route ([type]): [description]
    """
    pids.loc[:, 'closest_eid'] = pids.apply(lambda x: route.loc[route.distance(x.geometry).idxmin()].eid , axis=1)

    # TODO add the father node of the `traj ID`
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
    df_mapping = df_mapping.merge(gdf_panos['MoveDir'], left_on='pid', right_index=True)

    df_mapping.loc[:, 'offset'] = df_mapping.apply(
        lambda x: cal_relative_offset(gdf_panos.loc[x.pid].geometry, net.df_edges.loc[x.name].geom_origin)[0] / net.df_edges.loc[x.name].dist, 
        axis=1
    )
    df_mapping = df_mapping.reset_index().rename(columns={'index': 'eid'}).sort_values(['eid', 'offset'])
    
    if df_pred_memo is not None:
        df_mapping = df_mapping.merge(df_pred_memo[['PID', 'lane_num', 'DIR']].rename(columns={"PID": 'pid', 'DIR':'MoveDir'}), on=['pid', 'MoveDir'])

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

        median = int(np.median(points.lane_num))
        remain_ponas_index = np.sort(points.index)[trim_nums: -trim_nums] if trim_nums != 0 else np.sort(points.index)

        tmp  = points[['offset','lane_num']]
        prev = points.lane_num.shift(-1) == points.lane_num
        nxt  = points.lane_num.shift(1) == points.lane_num
        not_continuous = tmp[(prev|nxt) == False].offset.values.tolist()
        
        idxs = points.query( f"not (offset not in {not_continuous} \
                        and index in @remain_ponas_index)", 
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
    points_none_pred = points.query('lane_num<=0')
    points_none_pred.loc[:, 'outlier'] = True
    points.query('lane_num > 0', inplace=True)
    
    points = points.groupby('eid').apply( _panos_outlier )
    points = points.groupby('eid').apply( _panos_filter )
    
    return points.append(points_none_pred).sort_values(['eid', 'offset'])


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

def matching_edge_helper(queue, uf, eids=[], coverage_thred=.7, logger=None):
    """[summary]

    Args:
        queue ([type]): [description]
        uf ([type]): [description]
        eids ([type], optional): Define the whole eids needed to be matched. Defaults to None.
        coverage_thred (float, optional): [description]. Defaults to .7.
        logger ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    global MATCHING_MEMO, eid_visited, rid_visited

    def _check_eid_visited(eid):
        if eid in eid_visited:
            if logger is not None:
                logger.debug(f"{eid}, {rid}, eid had been visited")
            return True
        
        ori_coverage = coverage_dict.get(eid, None) 
        if ori_coverage is not None and ori_coverage['percentage'] > coverage_thred:
            if logger is not None:
                logger.info(f"{eid} had been matched, ori_coverage rate: {ori_coverage['percentage']:.3f}")
            return True
        
        return False
    
    def _check_rid_visited(rid):
        if rid in rid_visited:
            logger.info(f"{eid}, {rid}, rid had been vistied")
            return True
        
        return False
    
    def _check_traj(traj):
        if traj is None:
            if logger is not None:
                logger.warning(f"{eid}, {rid} has no trajectory")
            return False
        
        return True
    
    def _update_edge_coverage_ratio(matching_res, rid, eid_new_visited=None, eids=[]):
        global MATCHING_MEMO
        edge_related_new = net.df_edges[['eid']].merge(matching_res['path'], on=['eid'])
        if edge_related_new.query(f"eid in @eids").shape[0] == 0:
            return None
        
        new_cover_dict = cal_coverage_helper(edge_related_new, format='dict')
        for key, val in new_cover_dict.items():
            if val['percentage'] > coverage_thred:
                if eid_new_visited is not None:
                    eid_new_visited.add(key)
                eid_visited.add(key)
            coverage_dict[key] = val
        new_coverage = coverage_dict[eid]['percentage'] if eid in coverage_dict else 0
        
        for rid in uf.get_traj(rid):
            rid_visited.add(rid)
        
        print(f'{eid}, {rid}, coverage rate: {new_coverage:.3f}')
        ST_MATCHING_DICT[ uf.father[rid]] = matching_res['path']
        map_pid_to_edge(traj, matching_res['path'], PID_TO_EDGE, EDGE_TO_PID)
        MATCHING_MEMO = pd.concat([MATCHING_MEMO, matching_res['path']])
        
        return new_coverage
    
    while queue:
        eid_visited_new = set()
        _, eid, rid = heapq.heappop(queue) 
        traj = uf.get_panos(rid, False)
        ori_coverage = coverage_dict.get(eid, None) 

        if _check_eid_visited(eid):
            break
        if _check_rid_visited(rid):
            continue
        if not _check_traj(traj):
            continue
        if logger is not None:
            ratio = ori_coverage['percentage'] if ori_coverage is not None else 0
            logger.info(f"{eid}({ratio:.2f}): {rid}")
            
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
        new_coverage = _update_edge_coverage_ratio(matching_res, rid, eid_visited_new, eids)
        if new_coverage is None or (ori_coverage is not None and new_coverage == ori_coverage['percentage']): # or len(eid_new_visited)==0
            if logger is not None:
                logger.info(f'\t not related traj, visited: {eid_visited_new}')
            continue
        
        # post process
        # print(f'{eid}, {rid}, coverage rate: {new_coverage:.3f}')
        # ST_MATCHING_DICT[rid] = matching_res['path']
        # map_pid_to_edge(traj, matching_res['path'], PID_TO_EDGE, EDGE_TO_PID)
        # MATCHING_MEMO = pd.concat([MATCHING_MEMO, matching_res['path']])

        if new_coverage <= coverage_thred and logger is not None:
            logger.info(f'\t visited: {eid_visited_new}')
        if new_coverage > coverage_thred:
            eid_visited_new.add(eid)
            eid_visited.add(eid)
            if logger is not None:
                logger.info(f'\t visited: {eid_visited_new}')
            break
    
    pass


def traverse_edges(edges, gdf_roads, uf, plot_candidates_rids=False, coverage_thred=.7, logger=None):
    """traverse edges

    Args:
        edges ([type]): [description]
        gdf_roads ([type]): Pano edge.
        coverage_thred (float, optional): [description]. Defaults to .7.
        plot_candidates_rids (bool, optional): [description]. Defaults to False.
        logger ([type], optional): [description]. Defaults to None.
    """
    def _rid_heap_helper(item, net):
        """create the heap item for road traverse.
        Args:
            item ([type]): [description]
            net ([type]): [description]
        Returns:
            [type]: [description]
        """
        record = {
            # TODO add similarity, 参考轨迹相似度编写
            'dis': gdf_panos.query(f"RID==@item.name").distance(net.df_edges.loc[item['eid']].geometry).mean() / DIS_FACTOR,
            'eid': item['eid'],
            'rid': item.name,
            'rid_father': uf.find(item.name)
        }

        return record

    def _get_related_rids_by_edge_buffer(df_edges_unvisted, gdf_roads, dis_buffer=25):
        """Get related pano rids by edge buffer. 

        Args:
            df_edges_unvisted ([type]): [description]
            gdf_roads ([type]): [description]
            dis_buffer (int, optional): [description]. Defaults to 25.

        Returns:
            [type]: [description]
        """    
        road_mask = gpd.GeoDataFrame({
            'eid': df_edges_unvisted['eid'].values, 
            'geometry':df_edges_unvisted.buffer(dis_buffer*DIS_FACTOR)}
        )
        related_rids = gpd.sjoin(gdf_roads, road_mask, op='intersects')

        return related_rids

    eids_lst = edges.eid.unique().tolist()
    related_rids = _get_related_rids_by_edge_buffer(edges, gdf_roads)

    for idx in range(len(eids_lst)):
        rids = related_rids.query(f'eid=={eids_lst[idx]}')
        if rids.shape[0] == 0:
            continue
        
        queue = rids.apply(lambda x: _rid_heap_helper(x, net), axis=1, result_type='expand').\
                     sort_values('dis').groupby("rid_father").head(1).\
                     sort_values('dis')[['dis', 'eid', 'rid']].values.tolist()

        if len(queue) == 0 or eids_lst[idx] in eid_visited:
            continue

        if plot_candidates_rids:
            plot_neighbor_rids_of_edge(eids_lst[idx], edges)
        if logger is not None:
            logger.debug(f"queue: \n{queue}")
            
        matching_edge_helper(queue, uf, eids_lst, coverage_thred=coverage_thred, logger=logger)


def predict_for_area():
    
    coverage_dict = {}
    eid_visited = set()
    rid_visited = set()
    MATCHING_MEMO = gpd.GeoDataFrame(columns=['eid'])
    ST_MATCHING_DICT = {}
    PID_TO_EDGE = {}
    EDGE_TO_PID = {}

    bbox = box(114.05059,22.53084, 114.05887,22.53590)

    df_trajs, uf, edges = get_pano_topo_by_road(net, gdf_base, gdf_panos, clip_geom=bbox)
    edges.loc[:, 'link_level'] = edges.road_type.apply(lambda x: link_type_no_dict[x] if x in link_type_no_dict else 50)
    edges.sort_values('link_level', inplace=True)

    traverse_edges(edges[edges.link_level==7], gdf_roads, uf, logger=logger)


    df_pid_2_edge = pids_filter(
        sort_pids_in_edge(EDGE_TO_PID, df_pred_memo, plot=False)
    )

    df_lane_nums = pd.DataFrame(
        df_pid_2_edge[~df_pid_2_edge.outlier]\
            .groupby('eid')['lane_num']\
            .apply(list)\
            .apply(_get_lst_mode)
    )

    map_visualize( net.df_edges.merge(df_lane_nums, on='eid') )

    tmp = net.df_edges.merge(df_lane_nums, on='eid')
    tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'].astype(np.int)
    gdf_to_postgis(tmp, 'test_matching')


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


def _get_lst_mode(lst, offset=1):
    # FIXME: mode([5, 4, 4, 5]) -> 4
    res = stats.mode(lst)
    if res[1][0] == 1:
        return None
    
    return res[0][0] - offset if res[0][0] != 0 else None


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
""" initialize """
# ! new traverse road framework
coverage_dict = {}
eid_visited = set()
rid_visited = set()
MATCHING_MEMO = gpd.GeoDataFrame(columns=['eid'])
ST_MATCHING_DICT = {}
PID_TO_EDGE = {}
EDGE_TO_PID = {}

#%%

"""step 1: download OSM data"""
net = load_net_helper(bbox=SZ_BBOX, combine_link=True, reverse_edge=True, two_way_offeset=True, cache_folder='../../MatchGPS2OSM/cache')

"""step 2: dowload pano topo"""
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
pano_base_res = pano_base_main(project_name='futian', geom=futian_area)
gdf_base, gdf_roads, gdf_panos = pano_base_res['gdf_base'], pano_base_res['gdf_roads'], pano_base_res['gdf_panos']
# uf_all, _, _  = combine_rids(gdf_base, gdf_roads, gdf_panos, plot=False)
# map_visualize( pano_base_res['gdf_roads'], scale=.01 )

"""step 3: download pano imgs and predict"""
# pano_img_res = get_staticimage_batch(gdf_panos, 50, True)
gdf_panos = gdf_panos.merge(df_pred_memo[['PID', "DIR", 'lane_num']].rename(columns={"DIR": 'MoveDir'}), on=['PID', "MoveDir"], how='left').set_index('PID')
# panos预测所有的情况 -> drop_pano_file、get_staticimage_batch、lstr数据库中更新


#%%
coverage_dict = {}
eid_visited = set()
rid_visited = set()
MATCHING_MEMO = gpd.GeoDataFrame(columns=['eid'])
ST_MATCHING_DICT = {}
PID_TO_EDGE = {}
EDGE_TO_PID = {}

road_name = "中心一路"
# road_name = "滨河大道"
road_name = '滨河大道辅路'
# road_name = "福华三路"


df_trajs, uf, edges = get_pano_topo_by_road(net, gdf_base, gdf_panos, road_name=road_name, road_type=None, clip_geom=futian_area)

traverse_edges(edges, gdf_roads, uf, logger=logger)
df_lane_nums, edges = get_laneNum(edges, db_name = f'test_lane_res_{road_name}')

# %%
# DEBUG
# TODO check the None value

map_visualize( net.df_edges.merge(df_lane_nums, on='eid') )


# %%
bbox = box(114.05059,22.53084, 114.05887,22.53590)

edges_area = gpd.sjoin(
        net.df_edges,
        gpd.GeoDataFrame(geometry=[bbox]),
        op='intersects'
)

map_visualize(
    # net.df_edges[net.df_edges.within(bbox)]
    edges_area
)
#%%
