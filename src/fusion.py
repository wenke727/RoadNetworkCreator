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


#%%

df_pred_memo = load_df_memo(PRED_MEMO)

"""step 1: download OSM data"""
net = load_net_helper(bbox=SZ_BBOX, cache_folder='../../MatchGPS2OSM/cache')

"""step 2: dowload pano topo"""
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
pano_data = pano_base_main(project_name = 'futian', geom=futian_area)

"""step 3: download pano imgs and predict"""
# pano_img_res = get_staticimage_batch(gdf_panos, 50, True)
# panos预测所有的情况 -> drop_pano_file、get_staticimage_batch、lstr数据库中更新
pano_data['gdf_panos'] = pano_data['gdf_panos'].merge(
    df_pred_memo[['PID', "DIR", 'lane_num']].rename(columns={"DIR": 'MoveDir'}), on=['PID', "MoveDir"], how='left'
).set_index('PID')


#%%

def spatial_join(df, geom):
    tmp = gpd.sjoin(
            df,
            gpd.GeoDataFrame(geometry=[geom]),
            op='intersects'
    )
    
    return tmp


def map_pid_to_edge(root, pids, route, pid_to_edge, edge_to_pid ):
    """map pid to edge based on the result of map matching

    Args:
        pids ([type]): [description]
        route ([type]): [description]
    """
    pids.loc[:, 'closest_eid'] = pids.apply(lambda x: route.loc[route.distance(x.geometry).idxmin()].eid , axis=1)

    for i, item in pids.iterrows():
        eid = item['closest_eid']
        
        # format 0 
        # node = edge_to_pid[(eid, root)] = edge_to_pid.get((eid, root), [])
        # if item['PID'] not in node:
        #     node.append(item['PID'])
        
        # format 1
        node = edge_to_pid[eid] = edge_to_pid.get(eid, {})
        node[root] = node.get(root, [])
        if item['PID'] not in node[root]:
            node[root].append(item['PID'])
        
        pid_to_edge[item['PID']] = pid_to_edge.get(item['PID'], set())
        if eid not in pid_to_edge[item['PID']]:
            pid_to_edge[item['PID']].add(eid)

    return


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

    points.reset_index(drop=True, inplace=True)
    points.loc[:, 'outlier'] = False
    points_none_pred = points.query('lane_num<=0')
    points_none_pred.loc[:, 'outlier'] = True
    points.query('lane_num > 0', inplace=True)
    
    points = points.groupby('eid').apply( _panos_outlier )
    points = points.groupby('eid').apply( _panos_filter )
    
    return points.append(points_none_pred).sort_values(['eid', 'offset'])


def sort_pids_in_edge(edge_to_pid, gdf_panos, net, plot=False, filter=True):
    # TODO 针对有多个匹配方案的，选择距离最近的一个
    # df_mapping = pd.DataFrame([edge_to_pid]).T.rename(columns={0: "pid"}).explode('pid')
    df_mapping = pd.DataFrame(edge_to_pid).stack()\
                                        .reset_index()\
                                        .rename(columns={'level_0': 'rid', 'level_1': 'eid', 0: 'pid'})\
                                        [['eid','rid', 'pid']]\
                                        .explode('pid')\
                                        .reset_index(drop=True)
    df_mapping = df_mapping.merge(gdf_panos[['MoveDir', 'lane_num']], left_on='pid', right_index=True)
    df_mapping.loc[:, 'offset'] = df_mapping.apply(
        lambda x: cal_relative_offset(gdf_panos.loc[x.pid].geometry, net.df_edges.loc[x.eid].geom_origin)[0] / net.df_edges.loc[x.eid].dist, 
        axis=1
    )
    df_mapping.sort_values(['eid', 'offset'], inplace=True)
    
    if plot:
        map_visualize( gdf_panos.loc[ df_mapping.pid] )

    if filter:
        df_mapping = pids_filter(df_mapping)
    
    return df_mapping


class DataFushion():
    def __init__(self, net, pano_data, logger=None, pano_dis_thred=25):
        self.net = net
        self.load_panos_data(pano_data)
        self.pano_dis_thred = pano_dis_thred

        self.pid_to_edge = {}
        self.edge_to_pid = {}
        self.coverage_dict = {}
        self.eid_visited = set()
        self.rid_visited = set()
        self.st_matching_dict = {}
        # self.matching_memo = gpd.GeoDataFrame(columns=['eid'])
        
        self.logger = LogHelper(log_name='fusion.log').make_logger(level=logbook.INFO) if logger is None else logger

    
    def load_panos_data(self, pano_data):
        self.gdf_base  = pano_data['gdf_base']
        self.gdf_roads = pano_data['gdf_roads']
        self.gdf_panos = pano_data['gdf_panos']

    
    """helper"""
    def _get_lst_mode(self, lst, offset=1):
        # FIXME: mode([5, 4, 4, 5]) -> 4, 仅有一个数值的时候怎么处理
        res = stats.mode(lst)
        if res[1][0] == 1:
            return None
        
        return res[0][0] - offset if res[0][0] != 0 else None


    def _plot_neighbor_rids_of_edge(self, eid, df_edges_unvisted, dis_buffer=25):
        """Plot the neighbor rids of the eid.

        Args:
            eid ([type]): [description]
            df_edges_unvisted ([type]): [description]
        """
        road_mask = gpd.GeoDataFrame({'eid': eid, 'geometry': df_edges_unvisted.query(f'eid=={eid}').buffer(dis_buffer*DIS_FACTOR)})
        tmp = gpd.sjoin(self.gdf_roads, road_mask, op='intersects')
        tmp.reset_index(inplace=True)
        
        fig, ax = map_visualize(tmp)
        tmp.plot(column='ID', ax=ax, legend=True)
        df_edges_unvisted.query(f'eid=={eid}').plot(ax=ax, color='blue', linestyle='--')
        
        return


    def _breakpoint_to_interval(self, item):
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


    def _get_df_lane_nums(self):
        df = pd.DataFrame(
            self.df_pid_2_edge[~self.df_pid_2_edge.outlier]\
                .groupby('eid')['lane_num']\
                .apply(list)\
                .apply(self._get_lst_mode)
        ) 
        
        return df


    def upload(self, db_name='test_matching'):
        tmp = net.df_edges.merge(self.df_lane, on='eid')
        tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'].astype(np.int)
        gdf_to_postgis(tmp, db_name)
        
        return True


    """ filter panos """
    def identify_edge_type(self, id, dis_thres=30):
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
        edge = self.net.df_edges.loc[id]
        if type(net.node[edge.e]['traffic_signals'])==str:
            return 3
        
        if type(net.node[edge.s]['traffic_signals'])==str:
            return 4

        if self.net.edge.get((edge.e, edge.s), 1000) > dis_thres:
            return 0
        
        path = net.a_star(edge.e, edge.s)
        if path is None or path['path'] is None:
            return 0
        
        if len(path['path']) == 2 and edge.dist < dis_thres and path['cost'] < dis_thres*2:
            return 1
        
        if len(path['path']) == 4 and edge.dist < dis_thres and path['cost'] < dis_thres*4:
            return 2
        
        return 0


    def filter_panos_by_road(self, road_name=None, road_type='primary', filter_sql=None, clip_geom=None):
        df_edges = self.net.df_edges.copy()
        if road_name is not None:
            df_edges = df_edges.query("name == @road_name")
        if road_name is None and road_type is not None:
            df_edges = df_edges.query("road_type == @road_type")
        if filter_sql is not None:
            df_edges.query(filter_sql, inplace=True)
        if clip_geom is not None:
            df_edges = df_edges.loc[ gpd.clip(df_edges, clip_geom).index ]
            
        road_mask = gpd.GeoDataFrame(df_edges.buffer(self.pano_dis_thred * DIS_FACTOR), columns=['geometry'])
        road_mask.loc[:, 'att'] = 1
        mask = road_mask.dissolve('att')

        idxs = gpd.clip(self.gdf_roads, mask.iloc[0].geometry, keep_geom_type=True).index
        
        return self.gdf_roads.loc[idxs], df_edges


    def get_pano_topo_by_road(self, road_name=None, road_type=None, filter_sql=None, clip_geom=None, with_level=True):
        """get pano topo by level/sql/geom.
        """
        roi, eoi = self.filter_panos_by_road(road_name=road_name, road_type=road_type, filter_sql=filter_sql, clip_geom=clip_geom)
        eoi.loc[:, 'edge_type'] = eoi.eid.apply(self.identify_edge_type)

        pids = np.unique(roi.src.tolist() + roi.dst.tolist()).tolist()
        uf, df_topo, df_topo_prev = combine_rids(self.gdf_base.loc[pids], roi, self.gdf_panos, plot=False, logger=self.logger)
        
        if with_level:
            eoi.loc[:, 'road_level'] = eoi.road_type.apply(lambda x: link_type_no_dict[x] if x in link_type_no_dict else 50)
            eoi.sort_values('road_level', inplace=True)

        return uf, eoi


    """ traverse """
    def traverse_edges(self, edges, uf, plot_candidates_rids=False, coverage_thred=.7):
        """traverse edges

        Args:
            edges ([type]): [description]
            gdf_roads ([type]): Pano edge.
            coverage_thred (float, optional): [description]. Defaults to .7.
            plot_candidates_rids (bool, optional): [description]. Defaults to False.
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
                'dis': self.gdf_panos.query(f"RID==@item.name").distance(net.df_edges.loc[item['eid']].geometry).mean() / DIS_FACTOR,
                'eid': item['eid'],
                'rid': item.name,
                'rid_father': uf.find(item.name),
            }

            return record

        def _get_related_rids_by_edge_buffer(df_edges_unvisted, dis_buffer=25):
            """Get related pano rids by edge buffer. 

            Args:
                df_edges_unvisted ([type]): [description]
                dis_buffer (int, optional): [description]. Defaults to 25.

            Returns:
                [type]: [description]
            """    
            road_mask = gpd.GeoDataFrame({
                'eid': df_edges_unvisted['eid'].values, 
                'geometry':df_edges_unvisted.buffer(dis_buffer*DIS_FACTOR)}
            )
            related_rids = gpd.sjoin(self.gdf_roads, road_mask, op='intersects')

            return related_rids

        eids_lst = edges.eid.unique().tolist()
        related_rids = _get_related_rids_by_edge_buffer(edges)

        for eid in eids_lst:
            if eid in self.eid_visited:
                continue
            
            rids = related_rids.query(f'eid=={eid}')
            if rids.shape[0] == 0:
                continue
            
            tmp = rids.apply(lambda x: _rid_heap_helper(x, net), axis=1, result_type='expand').\
                       sort_values('dis').groupby("rid_father").head(1)
            tmp.loc[:, 'traj'] = tmp.rid.apply(uf.get_panos)
            tmp.query(f'traj == traj and rid not in {list(self.rid_visited)}', inplace=True)
            
            # queue = tmp.sort_values('dis')[['dis', 'eid', 'rid', 'traj']].values.tolist()
            queue = tmp[['dis', 'eid', 'rid', 'traj']].values.tolist()

            if len(queue) == 0:
                continue

            if plot_candidates_rids:
                self._plot_neighbor_rids_of_edge(eid, edges)
            
            self.logger.debug(f"queue: \n{queue}")
            self.matching_edge_helper(queue, uf, eids_lst, coverage_thred=coverage_thred)


    def matching_edge_helper(self, queue, uf, eids=[], coverage_thred=.7):
        """[summary]

        Args:
            queue ([type]): [description]
            uf ([type]): [description]
            eids ([type], optional): Define the whole eids needed to be matched. Defaults to None.
            coverage_thred (float, optional): [description]. Defaults to .7.

        Returns:
            [type]: [description]
        """
        def _check_eid_visited(eid):
            if eid in self.eid_visited:
                self.logger.debug(f"{eid}, {rid}, eid had been visited")
                return True
            
            ori_coverage = self.coverage_dict.get(eid, None) 
            if ori_coverage is not None and ori_coverage['percentage'] > coverage_thred:
                self.logger.info(f"{eid} had been matched, ori_coverage rate: {ori_coverage['percentage']:.3f}")
                return True
            
            return False
        
        def _check_rid_visited(rid):
            if rid in self.rid_visited:
                self.logger.info(f"{eid}, {rid}, rid had been vistied")
                return True
            
            return False
        
        def _check_traj(traj):
            if traj is None:
                self.logger.warning(f"{eid}, {rid} has no trajectory")
                return False
            
            return True
        
        def _update_edge_coverage_ratio(matching_res, rid, eid_new_visited=None, eids=[]):
            edge_related_new = net.df_edges[['eid']].merge(matching_res['path'], on=['eid'])
            if edge_related_new.query(f"eid in @eids").shape[0] == 0:
                return None
            
            new_cover_dict = self.cal_coverage_helper(edge_related_new, format='dict')
            for key, val in new_cover_dict.items():
                if val['percentage'] > coverage_thred:
                    if eid_new_visited is not None:
                        eid_new_visited.add(key)
                    self.eid_visited.add(key)
                self.coverage_dict[key] = val
            new_coverage = self.coverage_dict[eid]['percentage'] if eid in self.coverage_dict else 0
            
            for rid in uf.get_traj(rid):
                self.rid_visited.add(rid)
            
            print(f'{eid}, {rid}, coverage rate: {new_coverage:.3f}')
            traj_id = uf.father[rid]
            self.st_matching_dict[traj_id] = matching_res['path']
            map_pid_to_edge(traj_id, traj, matching_res['path'], self.pid_to_edge, self.edge_to_pid)
            # self.matching_memo = pd.concat([self.matching_memo, matching_res['path']])
            
            return new_coverage
        
        while queue:
            eid_visited_new = set()
            _, eid, rid, traj = heapq.heappop(queue) 
            # traj = uf.get_panos(rid, False)
            ori_coverage = self.coverage_dict.get(eid, None) 

            # if _check_eid_visited(eid):
            #     break
            # if _check_rid_visited(rid):
            #     continue
            # if not _check_traj(traj):
            #     continue

            ratio = ori_coverage['percentage'] if ori_coverage is not None else 0
            log_info = f"{eid}({ratio:.2f}) {rid}"
            # self.logger.info(f"{eid}({ratio:.2f}): {rid}")
                
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
            if new_coverage is None or (ori_coverage is not None and new_coverage == ori_coverage['percentage']):
                self.logger.info(f'{log_info}, not related traj, visited: {eid_visited_new}')
                continue

            if new_coverage <= coverage_thred:
                self.logger.info(f'{log_info}, visited: {eid_visited_new}')
            else:
                eid_visited_new.add(eid)
                self.eid_visited.add(eid)
                self.logger.info(f'{log_info}, visited: {eid_visited_new}')
                break
        
        pass


    def cal_coverage_helper(self, df, coverage_thred=.6, format='dataframe'):
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
        df_.loc[:, 'intervals'] = df_.apply(self._breakpoint_to_interval, axis=1)
        
        if df_.shape[0] == 2 and df_.eid.nunique() == 1:
            self.logger.debug(f"检查针对起点和终点都是同一个路段的情况，区间是否会有错误，如：49691, 79749\n{df_}")
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


    def get_unvisited_edge(self, edges, plot=True):
        # TODO
        mathing_ = self.cal_coverage_helper(edges, format='dataframe')
        edges_ = edges.merge(mathing_, on=['eid'], how='left')
        edges_.visited = edges_.visited.fillna(False)
        
        if plot:
            map_visualize(edges_[~edges_.visited])
            
        return edges_[~edges_.visited]


    # main
    def predict_area_prepare(self, clip_geom):
        self.uf, self.df_eoi = self.get_pano_topo_by_road(clip_geom=clip_geom)


    def predict_area_start(self, lowest_road_level=6):
        eoi = self.df_eoi[self.df_eoi.road_level <= lowest_road_level]
        self.traverse_edges(eoi, self.uf)
        
        print('start df_pid_2_edge ...')
        self.df_pid_2_edge = sort_pids_in_edge(self.edge_to_pid, self.gdf_panos, self.net)
        self.df_lane = self._get_df_lane_nums()
        
        return eoi


#%%
# DEBUG
def debug_one_edge_mutli_traj(self:DataFushion):
    # ! 优化针对不同的轨迹，匹配的结果分隔开的情景
    """用于检查1个路段，多个匹配结果的情况

    Args:
        self (DataFushion): [description]
    """
    rid_0 = self.uf.father['01c473-32d2-a493-b92a-06ce69']
    rid_1 = self.uf.father['3dfb05-5f06-ba42-fc8b-f15d26']

    traj_0 = self.uf.get_panos(rid_0, plot=True).copy()
    traj_1 = self.uf.get_panos(rid_1, plot=True).copy()

    edge_0 = self.st_matching_dict[rid_0].copy()
    edge_1 = self.st_matching_dict[rid_1].copy()


    pid_to_edge = {}
    edge_to_pid = {}

    map_pid_to_edge(rid_0, traj_0, edge_0, pid_to_edge, edge_to_pid)
    map_pid_to_edge(rid_1, traj_1, edge_1, pid_to_edge, edge_to_pid)

    edge_to_pid


    # case: 一个路段对应多个匹配结果
    self.edge_to_pid[53549]
    self.edge_to_pid[28959]


    return



#%%
""" step 1: initiale """
geom = box(114.05059,22.53084, 114.05887,22.53590)
self = DataFushion(net=net, pano_data=pano_data)
self.predict_area_prepare(geom)

#%%
""" step 2: predict """
eoi = self.predict_area_start(lowest_road_level=5)
map_visualize(eoi)
self.upload()
# gdf_to_postgis(eoi.merge(self.df_lane, on='eid', how='left'), 'test_matching_all')


# eoi = eoi[eoi.road_level <= 5]

#%%
# TODO 找出所有没有识别的 车道

def get_edge_pano_lane(self):
    df = self.df_pid_2_edge

    df1 = pd.DataFrame(
        df[~df.outlier]\
            .groupby('eid')['lane_num']\
            .apply(list)\
    )
    df1.loc[:, 'status'] = 1

    df2 = pd.DataFrame(
        df[df.outlier]\
            .groupby('eid')['lane_num']\
            .apply(list)\
    )
    df2.loc[:, 'status'] = 0

    tmp = pd.concat([df1, df2])
    
    return tmp


mathing_pano_lane = get_edge_pano_lane(self)
mathing_pano_lane.loc[73666]


# filter the unpredicted edge
idxs = list(np.setdiff1d( eoi.index, self.df_lane.index))

map_visualize( eoi.loc[idxs], scale=.1)

df_miss = eoi.loc[idxs].sort_values(['road_level', 'name', 'eid', 'edge_type'])

df_miss_link = df_miss.query("road_type.str.contains('link')", engine='python')
map_visualize(df_miss_link, scale=.1)


atts = ['eid', 'rid', 'name', 's', 'e', 'order', 'road_type', 'dir', 'lanes', 'dist', 'edge_type', 'road_level']
df_miss.query("not road_type.str.contains('link')", engine='python')[atts]

