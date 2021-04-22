#%%
import os
import sys
import copy
import pyproj
import pickle
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import geopandas as gpd
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString
import warnings
from collections import deque
warnings.filterwarnings('ignore')

sys.path.append("../src")
from road_network import OSM_road_network
from road_matching import *
# from road_matching import _matching_panos_path_to_network, get_panos_of_road_and_indentify_lane_type_by_id, df_edges, DB_panos, DB_roads
from utils.geo_plot_helper import map_visualize

from osm_helper import osm_get, osm_parser, add_coords_to_osm_node_hash, tranfer_to_sumo
from xml_helper import indent, update_element_attrib, print_elem

from utils.log_helper import LogHelper, logbook

g_log_helper = LogHelper(log_dir="/home/pcl/traffic/RoadNetworkCreator_by_View/log",log_name='sumo.log')
SUMO_LOG = g_log_helper.make_logger(level=logbook.INFO)


OSM_CRS = None
OSM_MATCHING_MEMO = {}

#%%
class SumoNet(object):
    def __init__(self, name, verbose=False, *args):
        global OSM_CRS
        self.name = name
        self.verbose = verbose
        self.edge_file = f'./{name}/{name}.edg.xml'
        self.edge_tree = ET.parse(self.edge_file)
        self.edge_root = self.edge_tree.getroot()
        # notice key in `key_to_edge` is string instead of long
        self.key_to_edge = {}
        
        self.node_file = f'./{name}/{name}.nod.xml'
        self.node_tree = ET.parse(self.node_file)
        self.node_root = self.node_tree.getroot()
        # notice key in `key_to_node` is string instead of long
        self.key_to_node = {}
        self.crs = CRS(self.node_root.findall('location')[0].get('projParameter')).to_epsg()
        OSM_CRS = self.crs
        self.edge, self.node = self.parse_xml()
        
    def parse_xml(self,):
        def _id_parser(x):
            flag = ""
            if x[0] == "-":
                flag = '-'
                x = x[1:]
            tmp =  int(x.split("#")[0].split('-')[0] if x.split("#")[0].split('-')[0] != '' else -1 )
            
            return -tmp if flag == '-' else tmp
        
        def _order_parser(x):
            if "#" not in x:
                return 0
            tmp = x.split("#")[-1]
            
            return int(tmp.split('-')[0]) if '-' in tmp else int(tmp)    
        
        edges, nodes = [], []
        for item in self.edge_tree.findall('edge'):
            self.key_to_edge[item.get('id')] = item
            
            info = {key: item.get(key) for key in  item.keys()}
            for p in item.findall('param'):
                info[p.get('key')] = p.get('value')
            edges.append(info)

        edges = pd.DataFrame(edges)
        edges.loc[:, 'rid'] = edges.id.apply( _id_parser )
        edges.loc[:, 'order'] = edges.id.apply( _order_parser )        
            
        for item in self.node_root.findall('node'):
            self.key_to_node[item.get('id')] = item
            
            nodes.append({key: item.get(key) for key in item.keys()})
        nodes = gpd.GeoDataFrame(nodes)
        nodes.loc[:, 'geometry'] = nodes.apply( lambda x: Point( float(x.x), float(x.y) ) ,axis=1 )
        nodes.set_crs(epsg=self.crs, inplace=True)

        return edges, nodes 

    def save(self):
        indent(self.edge_root)
        indent(self.node_root)
        self.node_tree.write(f'./{name}/{name}.nod.xml', encoding='utf-8')
        self.edge_tree.write(f'./{name}/{name}.edg.xml', encoding='utf-8')  

    def check_node(self, pid, osm_nodeHash, add=True):
        if str(pid) in self.key_to_node:
            return True
        
        if add: 
            self.add_node(pid, osm_nodeHash)
            return True
        
        return False    
    
    def add_node(self, pid, osm_nodeHash):
        x, y = osm_nodeHash[int(pid)]['coords']
        info = {"id": str(pid), 'x': str(x), 'y': str(y)}
        
        node = ET.Element('node', info)
        self.node_root.append(node)
        self.key_to_node[str(pid)] = node
        # TODO add record to dataframe
        
        if self.verbose: 
            print(f'insert node into SUMO node xml file:\n\t{info}\n')

        pass
    
    def remove_node(self, item):
        pass

    def parse_edge_elem(self, elem):
        info = {key: elem.get(key) for key in elem.keys()}
        idx = info.get('id')
        rid, order = idx.split("#")
        info['rid'], info['order'] = int(rid), int(order)
        for p in elem.findall('param'):
            info[p.get('key')] = p.get('value')        
        
        return info

    def add_edge(self, elem):
        """add `edge` element to tree
        
        Args:
            elem (Element): edge element
        """
        if elem.get('from') != elem.get('to'):
            self.edge_root.append(elem)
            info = self.parse_edge_elem(elem)
            
            self.edge = self.edge.append(info, ignore_index=True)
            self.key_to_edge[elem.get('id')] = elem

        else:
            print("not add to sum_net", "*"*30)
            print_elem(elem)
            
        pass

    def remove_edge_by_rid(self, rid):
        # rid = elem.get('id')
        if rid not in self.key_to_edge:
            return False
        
        edge_elem = self.key_to_edge[rid]
        self.edge_root.remove( edge_elem )
        del self.key_to_edge[rid]
        self.edge.query(f" id!='{rid}' ", inplace=True)
        
        if self.verbose:
            print("remove_edge_by_rid: ", rid)
        
        return True
        
    def update_edge_df(self, elem):
        rid = elem.get('id')
        size = self.edge.shape[0]
        self.edge.query(f" id!='{rid}' ", inplace=True)
        if self.verbose:
            print("\ndrop origin record", rid, f"{size} -> {self.edge.shape[0]}")    
        
        info = self.parse_edge_elem(elem)
        self.edge = self.edge.append(info, ignore_index=True)

        if self.verbose:
            print("update_edge_df success")
        
        return
    
    def get_edge_df_by_rid(self, rid):
        return self.edge.query(f"rid=={rid}").sort_values('order', ascending=True if rid > 0 else False)
    
    def get_edge_elem_by_id(self, rid):
        if rid not in self.key_to_edge:
            return None
        
        return self.key_to_edge[rid]

class MatchingPanos():
    def __init__(self, memo={}, cache_folder="../cache", *args):
        self.memo = memo
        self.cache_folder = cache_folder
        if self.cache_folder is not None:
            self.load_memo()
    
    def load_memo(self):
        if not os.path.exists(f'{self.cache_folder}/MatchingPanos_MEMO.pkl'):
            return
        
        self.memo = pickle.load( open(f'{self.cache_folder}/MatchingPanos_MEMO.pkl', 'rb') )
        print("MatchingPanos loading memo.pkl success!")
        return True
    
    def save_memo(self):
        self.memo = pickle.dump(self.memo, open(f'{self.cache_folder}/MatchingPanos_MEMO.pkl', 'wb'))
        
        return

    def add_lst(self, lst, df_edges, vis=False):
        for i in lst:
            self.add(i, vis)

        return
    
    def add(self, i, vis=False):
        if i in self.memo:
            return
        
        self.memo[i] = self.memo.get(i, {})
        df = get_and_filter_panos_by_osm_rid(i, vis)
        self.memo[i]['df'] = df
        self.memo[i]['median'] = int(df.lane_num.median())
        
        return
    
    def gdf_to_file_by_rid(self, rid, folder=None):
        """save the mathching record in the file

        Args:
            rid ([type]): [description]
            folder (dir): the path to store the dataframe if not `None` 

        Returns:
            [type]: [description]
        """
        if rid not in self.memo:
            print('please check the rid in the road set or not')
            return None
        
        df = self.memo[rid]['df']
        df.loc[:, 'RID'] = df.loc[:, 'RID'].astype(str)
        df.reset_index(inplace=True)
        
        if folder is not None:
            df.to_file( os.path.join(folder, f'{rid}.geojson'), driver="GeoJSON")
        
        return df

    @property
    def size(self):
        return len(self.memo)
    
    
# function 
def _get_revert_df_edges(road_id, df_edges, vis=False):
    """create the revert direction edge of rid in OSM file

    Args:
        road_id ([type]): the id of road
        df_edges ([type]): gdf create by 
        vis (bool, optional): plot the process or not. Defaults to False.

    Returns:
        [gdf]: the geodataframe of revert edge
    """
    road_id = road_id if road_id > 0 else -road_id
    df_tmp = df_edges.query(f"rid == {road_id} ")

    df_tmp.rid = -df_tmp.rid
    df_tmp.loc[:, ['s','e']] = df_tmp.loc[:, ['e','s']].values
    df_tmp.loc[:, 'index'] = df_tmp['index'].max() - df_tmp.loc[:, 'index']
    df_tmp.loc[:, 'geometry'] = df_tmp.geometry.apply( lambda x: LineString(x.coords[::-1]) )
    df_tmp.loc[:, 'pids'] = df_tmp.pids.apply( lambda x: ";".join( x.split(';')[::-1] ) )
    df_tmp.sort_values(by='index', inplace=True)
    # gpd.GeoDataFrame(pd.concat( [df_edges.query(f"rid == {road_id} "), df_tmp] )).to_file('./test.geojson', driver="GeoJSON")

    if vis:
        matching0 = get_panos_of_road_and_indentify_lane_type_by_id(-road_id, df_tmp, False)
        matching1 = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False)
        _, ax = map_visualize(matching0, scale =0.001)
        matching1.plot(column='level_0', legend=True, ax=ax, cmap='jet')
        matching0.plot(column='level_0', legend=True, ax=ax, cmap='jet')

    return df_tmp

def _panos_filter(panos):
    if panos.shape[0] == 2 and panos.lane_num.nunique() == 1:
        return panos

    remove_pano_num = 1
    median = int(np.median(panos.lane_num))
    remain_ponas_index = np.sort(panos.Order.unique())[remove_pano_num: -remove_pano_num]

    tmp = panos[['Order','lane_num']]
    prev = panos.lane_num.shift(-1) == panos.lane_num
    nxt = panos.lane_num.shift(1) == panos.lane_num
    not_continuous = tmp[(prev|nxt) == False].Order.values.tolist()
    
    panos.query( f" Order not in {not_continuous} \
                    and Order in @remain_ponas_index \
                    and abs(lane_num - @median) < 2", 
                    inplace=True 
                )
    
    return panos

def get_and_filter_panos_by_osm_rid(road_id = 243387686, vis=False, offset=1):
    """[summary]

    Args:
        road_id (int, optional): [description]. Defaults to 243387686.
        vis (bool, optional): [description]. Defaults to False.
        offset (int, optional): [the attribute `lane_num` is the real lane num or the real lane line num. If `lane_num` represent line num, then offset is 1. Other vise, the offset is 0 ]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    atts = ['index', 'RID', 'Name', 'geometry', 'lane_num', 'frechet_dis', 'angel', 'osm_road_id', 'osm_road_index', 'related_pos', 'link']
    if road_id > 0:
        matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False) 
        matching = matching[atts].merge(df_edges[['s', 'e']], left_on='osm_road_index', right_index=True)
    else:
        # FIXME -208128058 高新中三道, 街景仅遍历了一遍。。。。
        df_tmp = _get_revert_df_edges(road_id, df_edges)
        matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_tmp, False) 
        # print("!!!!matching", matching)
        matching = matching[atts].merge(df_tmp[['s', 'e']], left_on='osm_road_index', right_index=True)
        
    # filter outlier -> 计算路段的统计属性
    rids = matching.RID.unique().tolist()
    points = DB_panos.query( f"RID in {rids}" ).dropna()

    rid_order = CategoricalDtype(matching.RID, ordered=True)
    tmp = points.groupby('RID').apply( lambda x: _panos_filter(x) ).drop(columns='RID').reset_index()
    tmp.loc[:, 'RID'] = tmp['RID'].astype(rid_order)
    tmp.sort_values(by=['RID', 'Order'], inplace=True)
    tmp.reset_index(drop=True, inplace=True)

    if offset:
        tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'] - 1
        
    if vis:
        _, ax = map_visualize(tmp, scale=.1, color='gray', figsize=(15, 15))
        tmp.loc[:, 'lane_num_str'] = tmp.loc[:, 'lane_num'].astype(str)
        tmp.plot(ax=ax, column='lane_num_str', legend=True)

    return tmp

def merge_intervals(intervals, start, end, height):
    if start is None or height ==0 or start == end: 
        return 

    if not intervals:
        intervals.append( [start, end, height] )
        return
    
    _, prev_end, prev_height = intervals[-1]
    if prev_height == height and prev_end == start:
        intervals[-1][1] = end

        return  
    intervals.append([start, end, height])

def insert_intervals(intervals, newInterval):
    res = []
    insertPos = 0
    newInterval = newInterval.copy()
    for interval in intervals:
        if interval[1] < newInterval[0]:
            res.append(interval)
            insertPos += 1
        elif interval[0] > newInterval[1]:
            res.append(interval)
        else:
            newInterval[0] = min(interval[0], newInterval[0])
            newInterval[1] = max(interval[1], newInterval[1])
            newInterval[2] = interval[2]
    
    res.insert(insertPos, newInterval)

    return res

def cal_dis_two_point(pid0, pid1):
    assert pid0 in osm_nodeHash and pid1 in osm_nodeHash, "check input"
    if 'coords' in osm_nodeHash[pid0]:
        # a, b = osm_nodeHash[pid0]['coords'], osm_nodeHash[pid1]['coords']
        # math.sqrt( (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) )
        dis = np.linalg.norm(np.array(osm_nodeHash[pid0]['coords']) - np.array(osm_nodeHash[pid1]['coords']))
    else:
        from haversine import haversine, Unit
        a = osm_nodeHash[pid0]['geometry']
        b = osm_nodeHash[pid1]['geometry']
        dis = haversine( a.coords[0][::-1], b.coords[0][::-1], unit=Unit.METERS )
        # a.distance( b ) * 110 *1000
    
    return dis

def get_road_changed_section(rid, vis=True, length_thres=20):
    """获取变化的截面

    Args:
        rid ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    def _lane_seg_intervals(lane_num_dict):
        intervals = []
        for start, end, num in [(i, i+1, int(lane_num_dict[i])) for i in lane_num_dict ]:
            merge_intervals(intervals, start, end, num)

        return intervals

    def _convert_interval_to_gdf(intervals):
        change_pids = gpd.GeoDataFrame(intervals, columns=['pano_idx_0', 'pano_idx_1', 'lane_num'])

        change_pids.loc[:, 'pano_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].geometry )
        change_pids.loc[:, 'pano_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].geometry )
        change_pids.loc[:, 'pano_id_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].PID )
        change_pids.loc[:, 'pano_id_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].PID )
        change_pids.loc[:, 'rid0'] = change_pids.pano_0.apply( lambda x: lines.loc[lines.distance(x).argmin()].start )
        change_pids.loc[:, 'rid1'] = change_pids.pano_1.apply( lambda x: lines.loc[lines.distance(x).argmin()].end )
        change_pids.loc[:, 'length'] = change_pids.apply( lambda x: x.pano_0.distance(x.pano_1)*110*1000, axis=1 )

        return change_pids

    panos = OSM_MATCHING_MEMO[rid]['df']
    segments = panos.query(" lane_num != @ panos.lane_num.median() ")
    # 注意区间：左闭右开
    intervals = _lane_seg_intervals(segments['lane_num'].to_dict())

    pids = osm_wayHash[rid]['points'] if rid > 0 else osm_wayHash[-rid]['points'][::-1]
    lines = gpd.GeoDataFrame([ {'index':i, 
                                'start': osm_nodeHash[pids[i]]['pid'],
                                'end': osm_nodeHash[pids[i+1]]['pid'],
                                'geometry': LineString( [osm_nodeHash[pids[i]]['xy'],
                                                        osm_nodeHash[pids[i+1]]['xy']] )} 
                                for i in range(len(pids)-1) ],
                            ).set_crs(epsg=4326)

    # second layer for filter
    change_pids = _convert_interval_to_gdf((intervals))
    change_pids.query("length != 0", inplace=True)
    
    attrs = ['pano_idx_0', 'pano_idx_1', 'lane_num']
    keep      = change_pids.query(f"length >= {length_thres}")[attrs].values.tolist()
    if len(keep) < 1:
        return None
    candidate = change_pids.query(f"length < {length_thres}")[attrs].values.tolist()

    for i in candidate:
        keep = insert_intervals(keep, i)

    intervals = [i for i in keep if i not in candidate]
    change_pids = _convert_interval_to_gdf((intervals))
    change_pids.loc[:, 'intervals'] = change_pids.apply( lambda x: [pids.index(x.rid0), pids.index(x.rid1)], axis=1 )
    
    return change_pids

def osm_road_segments_intervals(x, plst):
    def helpler(x):
        if 'cluster' in x:
            # print('cluster: ',x, [plst.index(int(i)) for i in x.split("_")[1:] if int(i) in plst ] )
            id = max( [plst.index(int(i)) for i in x.split("_")[1:] if int(i) in plst ] )
        else:
            id = plst.index(int(x))
        
        return id

    return [helpler(x['from']), helpler(x['to'])]

def lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst, log=SUMO_LOG):
    lane_num_lst = [i[2] for i in new_intervals]
    for index, elem in enumerate(elem_lst):
        elem.set('id', str(id_lst[index]))
        elem.set('shape', shape_lst[index])
        
        elem.set('numLanes', str(lane_num_lst[index]))
        cur_lane_elems = elem.findall('lane')
        cur_lane_elems_num = len(cur_lane_elems)
        if lane_num_lst[index] > cur_lane_elems_num:
            for lane_id in range(cur_lane_elems_num, lane_num_lst[index]):
                new_lane = copy.deepcopy(cur_lane_elems[-1])
                new_lane.set('index', str(lane_id))
                elem.append( new_lane )
        elif lane_num_lst[index] < cur_lane_elems_num:
            for lane_id in range(lane_num_lst[index], cur_lane_elems_num):
                elem.remove(cur_lane_elems[lane_id])
        
        if index != 0:
            elem.set('from', str(pids[new_intervals[index][0]]))
            for record in elem.findall('param'):
                if 'origFrom' == record.get('key'):
                    elem.remove(record)
                    print('removing origFrom')
                    
        if index != len(elem_lst) - 1:
            elem.set('to', str(pids[new_intervals[index][1]]))
            for record in elem.findall('param'):
                if 'origTo' == record.get('key'):
                    elem.remove(record)
                    print('removing origTo')
        
        if index != 0:
            sumoNet.add_edge(elem)
        else:
            sumoNet.update_edge_df(elem)
        
        if elem.get('from') == elem.get('to'):
            sumoNet.remove_edge_by_rid(elem.get('id'))
        
        # FIXME `from` and `to` has the same id
        if  elem.get('to') in elem.get('from') or elem.get('from') in elem.get('to'):
            status = sumoNet.remove_edge_by_rid(elem.get('id'))
            SUMO_LOG.info(f"Remove_edge_by_rid\n\t{elem.get('id')}: {status}")
        
def lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set, log=None, verbose=True):
    origin_start, origin_end = item.interval
    log_info = []
    log_info.append(f"LANE_CHANGE_PROCESS")
    log_info.append(f"\tsplit intervals {item.id}, origin: [{origin_start}, {origin_end}], insert: [{new_start}, {new_end}]")

    s = origin_start if pd.isnull(item.origFrom) else pids.index(int(item.origFrom))
    e = origin_end   if pd.isnull(item.origTo)   else pids.index(int(item.origTo))
    new_intervals = [[s        , new_start, int(item.numLanes)], 
                     [new_start, new_end  , lane_num_new], 
                     [new_end  , e        , int(item.numLanes)]
                    ]
    print(new_intervals)
    
    new_intervals = [[i,j,k] for i,j,k in new_intervals if i != j]
    print(new_intervals)
    # check the distance of last interval 
    last_seg_dis = cal_dis_two_point( pids[new_end], pids[new_intervals[-1][1]])
    if last_seg_dis < dis_thres:
        _, end, _ = new_intervals.pop()
        new_intervals[-1][1] = end
    log_info.append(f"\tnew_intervals: {new_intervals}, the last segment dis is {last_seg_dis}")
    
    shape_lst = []
    for s, e, _ in new_intervals:
        sumoNet.check_node(pids[e], osm_nodeHash)
        sumoNet.check_node(pids[s], osm_nodeHash)
        
        shape_tmp = " ".join( [",".join([ str(i) for i in osm_nodeHash[p]['coords']]) for p in pids[s:e+1]] )
        shape_lst.append(shape_tmp)
    
    id_lst = []
    cur_order = item.order
    order_lst = [cur_order]
    for i in range(len(new_intervals)-1):
        while cur_order in order_set:
            cur_order += 1
        order_set.add(cur_order)
        order_lst.append(cur_order)
    id_lst = [ f"{int(item.rid)}#{int(i)}"  for i in order_lst ]
    
    log_info.append(f"\tid: {id_lst}")
    log_info.append(f"\torder: {order_lst}")
    log_info.append(f"\tnew_intervals:{new_intervals}")
    log_info.append(f"\tshape_lst: {shape_lst}", )
    
    if log is not None:
        log.info( "\n".join(log_info)+"\n" )
    if verbose:
        for i in log_info:
            print(i)
    
    origin_edge = sumoNet.get_edge_elem_by_id(item.id)
    elem_lst = [origin_edge] + [copy.deepcopy(origin_edge) for _ in range(len(new_intervals)-1)]
    
    lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst)

    if verbose:
        for _, elem in enumerate(elem_lst):
            print_elem(elem, '\t')
                
def modify_road_shape(rid, dis_thres=25):
    # TODO write log
    change_pids = get_road_changed_section(rid)
    if change_pids is None:
        return
    
    pids = osm_wayHash[rid]['points'] if rid > 0 else osm_wayHash[-rid]['points'][::-1]
    road = sumoNet.get_edge_df_by_rid(rid)
    road.loc[:, 'interval'] = road.apply(lambda x: osm_road_segments_intervals(x, pids), axis=1)
    order_set = set( road.order.values )

    queue = deque( change_pids[['intervals', 'lane_num']].values.tolist() )
    while queue:
        [new_start, new_end], lane_num_new = queue.popleft()
        # print(f"new: [{new_start}, {new_end}], queue: {queue}")
    
        if new_start == new_end:
            continue
        
        for index, item in road.iterrows():
            origin_start, origin_end = item.interval
    
            if origin_start >= new_end:
                # print(f'\tbreak: {item.id}')
                break
            elif origin_end <= new_start:
                continue
            else:
                # origin: [30, 33], insert: [24, 32] 跨区间怎么处理？
                if origin_start > new_start and origin_start != new_end:
                    queue.appendleft([[origin_start, new_end], lane_num_new ])
                    queue.appendleft([[new_start, origin_start], lane_num_new ])
                    # print( f"\n!!!! origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                    break           
                
                if new_start > origin_start and new_end > origin_end:
                    queue.appendleft([[origin_end, new_end], lane_num_new ])
                    queue.appendleft([[new_start, origin_end], lane_num_new ])
                    # print( f"\n???? origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                    break
                
                # print( f"\n!!!! post origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set, SUMO_LOG)
                
                road = sumoNet.get_edge_df_by_rid(rid)
                road.loc[:, 'interval'] = road.apply(lambda x: osm_road_segments_intervals(x, pids), axis=1)
                break

def _pre_process_fine_tune(name, osm_file, log=False, SUMO_HOME="/usr/share/sumo"):
    """
    sumo releted process before fine tune
    """
    flag = False
    pre_process = f' rm -r ./{name}; mkdir {name}; cp {osm_file} ./{name}/{osm_file}; cd ./{name}; export SUMO_HOME={SUMO_HOME}'

    cmd = f"""
        {SUMO_HOME}/bin/netconvert  -t {SUMO_HOME}/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files {osm_file} --keep-edges.by-vclass passenger --offset.disable-normalization true -o {name}.net.xml
    """

    # create node, edge files
    cmd_tranfer0 = f"""{SUMO_HOME}/bin/netconvert --sumo-net-file {name}.net.xml --plain-output-prefix {name}; """
    for i in os.popen(' '.join( [pre_process, cmd, cmd_tranfer0] )).read().split('\n'): 
        if log: print(i)
        if 'Writing network' in i:
            flag = True
    
    return flag

def _post_process_fine_tune(name, osm_file, log=False, SUMO_HOME="/usr/share/sumo"):
    """
    sumo releted process post fine tune
    """

    flag = False
    sumoNet.save()
    
    post_precess = f"cd ./{name}; ll; cp ../start_with_net.sh ./; sh start_with_net.sh "
    res = os.popen(post_precess).read()
    if log: print(res)
    
    if 'calling /usr/share/sumo/bin/duarouter' in res:
        flag = True
    
            
    return flag




# fine_tune_road_set = [name_to_id['高新中四道']] + [road_id, -road_id]
fine_tune_road_set = [ name_to_id['科苑北路'] ]

SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'

tree = ET.parse('./osm_bbox.osm.bak.xml')
root = tree.getroot()
# 为osm里边的对象
osm_wayHash, osm_nodeHash = osm_parser(root)

"""粗调"""
for rid in fine_tune_road_set:
    if rid < 0:
        continue
    print(f'粗调: {rid}')
    lane = OSM_MATCHING_MEMO[rid]['median']
    item = osm_wayHash[rid]
    update_element_attrib(item['elem'], 'lanes', int(lane))

indent(root)
tree.write('osm_bbox.osm.xml', encoding='utf-8')
assert _pre_process_fine_tune(name, osm_file, False), 'check `_pre_process_fine_tune` functions'

sumoNet = SumoNet('osm')
add_coords_to_osm_node_hash(osm_nodeHash, OSM_CRS)

"""微调"""
for rid in fine_tune_road_set:
    modify_road_shape(rid)

assert _post_process_fine_tune(name, osm_file, False), 'check `_post_process_fine_tune` functions'




# %%
matchingPanos.gdf_to_file_by_rid(name_to_id['科苑北路'], './')



#%%
# 初始化
name_to_id = {'高新中四道': 529249851,
              '科技中二路': 208128052,
              '科苑北路': 231901939,
              }
rid = road_id = 208128052 # 科技中二路

RID_set = [ road_id, -road_id, 208128058, 529249851, -529249851, name_to_id['科苑北路']] # -208128058

matchingPanos = MatchingPanos()
matchingPanos.add_lst(RID_set, df_edges)
OSM_MATCHING_MEMO = matchingPanos.memo
# matchingPanos.save_memo()


#%%
# ! test for log


# %%
