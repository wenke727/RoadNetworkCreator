#%%
import os
import sys
import copy
import pickle
import datetime
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import geopandas as gpd
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString
import warnings
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.append("../src")
from road_network import OSM_road_network
from road_matching import *
from road_matching import _matching_panos_path_to_network, get_panos_of_road_and_indentify_lane_type_by_id, df_edges, DB_panos, DB_roads
from utils.geo_plot_helper import map_visualize
from utils.spatialAnalysis import relation_bet_point_and_line

from osm_helper import osm_get, osm_parser, add_coords_to_osm_node_hash, tranfer_to_sumo
from xml_helper import indent, update_element_attrib, print_elem
from interval_process import insert_intervals, merge_intervals

from utils.log_helper import LogHelper, logbook, log_type_for_sumo

g_log_helper = LogHelper(log_dir="/home/pcl/traffic/RoadNetworkCreator_by_View/log", log_name='sumo.log')
SUMO_LOG = g_log_helper.make_logger(level=logbook.INFO)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

OSM_CRS = None
OSM_MATCHING_MEMO = {}
matchingPanos = None

#%%

class OSM_Net():
    def __init__(self, file='./osm_bbox.osm.bak.xml', save_fn='./osm_bbox.osm.xml', logger=None):
        self.tree = ET.parse(file)
        self.root = self.tree.getroot()
        self.key_to_edge, self.key_to_node = self.osm_parser()
        self.road_level_dict = self.orginize_roads_based_grade()
        self.road_levels = self.road_level_dict.keys()
        self.save_fn = save_fn
        self.logger = logger
        # TODO add self.pids_memo
        self.pids_memo = {}
        
    def save(self, ):
        indent(self.root)
        self.tree.write(self.save_fn, encoding='utf-8')
        return True
    
    def orginize_roads_based_grade(self,):
        road_level_dict = {}
        for key, node in self.key_to_edge.items():
            if 'highway' not in node: 
                continue
            
            level = node['highway']
            if level not in road_level_dict:
                road_level_dict[level] = set([])
            road_level_dict[level].add(key)

        road_level_dict.keys()
        
        return road_level_dict

    def get_roads_by_road_level(self, level):
        if level not in self.road_level_dict:
            print(f'There is no {level} roads in the dataset')
            return False

        return list(self.road_level_dict[level])
    
    def osm_parser(self):
        wayHash, nodeHash = {}, {}
        for element in self.root.findall('way'):
            id = int(element.get('id'))
            if id not in wayHash:
                wayHash[id] = {}
                wayHash[id]['elem'] = element
                wayHash[id]['points'] = [ int(i.get('ref')) for i in element.findall('nd')]
                
                for i in element.findall('tag'):
                    wayHash[id][i.get('k')] = i.get('v')
                
        for node in self.root.findall('node'):
            id = int(node.get('id'))
            if id in nodeHash: continue

            info = {x.get('k'):x.get('v') for x in node.getchildren()} if node.getchildren() else {}
            info['pid'] = id
            info['xy'] = (float(node.get('lon')), float(node.get('lat')))
            info['geometry'] = Point( *info['xy'] )
            nodeHash[id] = info

        return wayHash, nodeHash

    def add_sumo_net_node_to_osm(self, sumo_net ):
        size = len(self.key_to_node)
        for i, item in sumo_net.node.to_crs(epsg=4326).iterrows():
            if item.id in self.key_to_node:
                continue

            self.key_to_node[item.id]  = { 'pid': item.id, 
                                            "xy": item.geometry.coords[0], 
                                            "geometry": item.geometry, 
                                            "coords": (float(item.x), float(item.y)) 
                                          }
        print(f"add_new_node_to_osm, node size {size} -> {len(self.key_to_node)}")
        
        return True

    def cal_dis_two_point(self, pid0, pid1):
        assert pid0 in self.key_to_node and pid1 in self.key_to_node, "check input"
        if 'coords' in self.key_to_node[pid0]:
            dis = np.linalg.norm(np.array(self.key_to_node[pid0]['coords']) - np.array(self.key_to_node[pid1]['coords']))
        else:
            from haversine import haversine, Unit
            a = self.key_to_node[pid0]['geometry']
            b = self.key_to_node[pid1]['geometry']
            dis = haversine( a.coords[0][::-1], b.coords[0][::-1], unit=Unit.METERS )
        
        return dis

    def add_coords_to_node(self, OSM_CRS=32649):
        """add projection coordinations of each node to osm node hash

        Args:
            key_to_node (dict): osm node converted from osm xml file 

        Returns:
            [type]: [description]
        """
        # assert OSM_CRS is not None, 'please process `parser_sumo_node_edge` to obtain `OSM_CRS`'
        df = gpd.GeoDataFrame(self.key_to_node).T
        df = df.set_crs(epsg=4326).to_crs(epsg=OSM_CRS)
        df.loc[:, 'coords'] = df.geometry.apply(lambda x: [ round(i,2) for i in x.coords[0]])

        for i in self.key_to_node.keys():
            self.key_to_node[i]['coords'] = df.loc[i, 'coords']
            
        return True

    def get_node_xy(self, node_id):
        if node_id not in self.key_to_node:
            return None
        return self.key_to_node[node_id]['xy']
        
    def get_node_coords(self, node_id):
        if node_id not in self.key_to_node:
            return None
        return self.key_to_node[node_id]['coords']
            
    def get_node_geometry(self, node_id):
        if node_id not in self.key_to_node:
            return None
        return self.key_to_node[node_id]['geometry']

    def get_pids_by_rid(self, rid, sumo_net, verbose=False, geo_plot=False):
        """get pids of a special road in a sumo net file

        Args:
            rid (int): the id of the road in the sumo net file
            sumo_net (Sumo_Net): the class of sumo net

        Returns:
            [lst]: the pids in order
        """
        assert isinstance(rid, int), f"Check the rid {rid} is availabel"
        if rid in self.pids_memo:
            return self.pids_memo[rid]
        # rid = 107586308， 存在和 `220885829`合并的情况； # 220885830 AddedRampe连续的情况
        pids = self.key_to_edge[rid]['points'] if rid > 0 else self.key_to_edge[-rid]['points'][::-1]
        # pids = osm_net.key_to_edge[rid]['points'] if rid > 0 else osm_net.key_to_edge[-rid]['points'][::-1]
        road_sumo = sumo_net.get_edge_df_by_rid(rid)
        
        if self.logger and verbose:
            attrs_show = ['id', 'from', 'to', 'numLanes', 'order']
            self.logger.info(road_sumo[attrs_show])
        
        if road_sumo.id.str.contains('Add').sum() == 0: 
            self.pids_memo[rid] = pids
            return pids

        insert_pids = {}
        for node1, node2 in road_sumo[['from', 'to']].values:
            node1 = int(node1) if node1.isdigit() else node1
            node2 = int(node2) if node2.isdigit() else node2

            if node1 not in pids:
                if node1 not in insert_pids:
                    insert_pids[node1] = {}
                insert_pids[node1]['next'] = node2
            
            if node2 not in pids:
                if node2 not in insert_pids:
                    insert_pids[node2] = {}
                insert_pids[node2]['prev'] = node1

        queue = deque( [(key, 
                         val['prev'] if 'prev' in val else '0', 
                         val['next'] if 'next' in val else '-1'
                         ) for key, val in insert_pids.items()] 
                      )

        while queue:
            node, prev, nxt = queue.popleft()
            prev = int(prev) if isinstance(prev, str) and prev.isdigit() else prev
            if nxt == '-1':
                nxt = pids[-1]
            else:
                nxt = int(nxt) if isinstance(nxt, str) and nxt.isdigit() else nxt

            prev_index = pids.index(prev) if prev in pids else 0
            nxt_index  = pids.index(nxt)  if nxt  in pids else len(pids) - 1

            for idx in range(prev_index, nxt_index):
                # print(f"intervals [{prev_index}, {nxt_index}], pids len {len(pids)}")
                flag = relation_bet_point_and_line(osm_net.get_node_coords(node), 
                                                   osm_net.get_node_coords(pids[idx]) + osm_net.get_node_coords(pids[idx+1]))
                if 0 <= flag <= 1:
                    break
            
            if flag <= 1:
                pids.insert(idx+1, node)
            else:
                pids += [node]

        if geo_plot:
            lst = pids
            gdf_nodes = gpd.GeoDataFrame(geometry=[osm_net.key_to_node[node]['geometry'] for node in lst])
            gdf_nodes.reset_index(inplace=True)

            fig, ax = map_visualize(gdf_nodes)
            gdf_nodes.plot( column='index', legend=True, ax=ax )
        
        self.pids_memo[rid] = pids
        
        return pids

    def get_edge_elem(self, rid):
        return self.key_to_edge[rid]['elem']        

    def rough_tune(self, lst, OSM_MATCHING_MEMO, save=True):
        for rid in tqdm(lst, 'process rough tune: '):
            if rid < 0: 
                continue

            print(f"\trough tune: {rid}, lane {OSM_MATCHING_MEMO[rid]['median'] if 'median' in OSM_MATCHING_MEMO[rid] else None } ")
            if 'median' in OSM_MATCHING_MEMO[rid] and OSM_MATCHING_MEMO[rid]['median'] is not None:
                update_element_attrib(self.get_edge_elem(rid), 'lanes', int(OSM_MATCHING_MEMO[rid]['median']))

        if save:
            self.save()

class Sumo_Net(object):
    def __init__(self, name, verbose=False, logger=None, *args):
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
        
        self.rids = set(self.edge.rid.unique())

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

    def check_node(self, pid, key_to_node, add=True):
        if str(pid) in self.key_to_node:
            return True
        
        if add: 
            self.add_node(pid, key_to_node)
            return True
        
        return False    

    def get_node(self, pid:str):
        if str(pid) not in self.key_to_node:
            return None
        return self.key_to_node[str(pid)]
    
    def add_node(self, pid, key_to_node):
        x, y = key_to_node[int(pid)]['coords']
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
        
        if "#" in idx:
            if '-' not in idx[1:]:
                rid, order = idx.split("#")
            else:
                rid, order = idx.split("#")
                order, postfix = order.split("-")
                info['postfix'] = postfix
            info['rid'], info['order'] = int(rid), int(order)
        else:
            rid, order = idx, 0           
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
            # TODO
            print("not add to sum_net", "*"*30)
            print_elem(elem)
            
        pass

    def remove_edge_by_rid(self, rid):
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

    def update_edge_elem_lane_num(self, id, lane_num):
        if id not in self.key_to_edge:
            return False
        
        lane_num = int(lane_num)
        elem = self.key_to_edge[id]
        elem.set('numLanes', str(lane_num))

        cur_lane_elems = elem.findall('lane')
        cur_lane_elems_num = len(cur_lane_elems)
        print('cur_lane_elems_num', cur_lane_elems_num, lane_num)

        if lane_num > cur_lane_elems_num:
            for lane_id in range(cur_lane_elems_num, lane_num):
                new_lane = copy.deepcopy(cur_lane_elems[-1])
                new_lane.set('index', str(lane_id))
                elem.append( new_lane )
        elif lane_num < cur_lane_elems_num:
            for lane_id in range(lane_num, cur_lane_elems_num):
                elem.remove(cur_lane_elems[lane_id])    
        print_elem(elem)
        self.edge.loc[ self.edge.id == id, 'numLanes'] = int(lane_num)
        print(self.edge.loc[self.edge.id == id, 'numLanes'])
        
        return True

    def get_edge_df_by_rid(self, rid):
        return self.edge.query(f"rid=={rid}").sort_values('order', ascending=True if rid > 0 else False)
    
    def get_edge_elem_by_id(self, rid, verbose=False):
        if rid not in self.key_to_edge:
            if verbose: print(f"{rid} not in sumo_net")
            return None
        
        item = self.key_to_edge[rid]
        if verbose: print_elem(item)
        return item

    def plot_edge(self, rid):
        tmp = self.edge.query( f"rid == {rid}" )['shape']
        tmp = tmp[~tmp.isna()]
        road_shape = tmp.apply( lambda x:  LineString( np.array( [ coord.split(',') for coord in  x.split(" ")]).astype(np.float) ))
        if road_shape.shape[0] == 0:
            return False
        
        road_shape = gpd.GeoDataFrame(geometry = road_shape, crs = OSM_CRS).to_crs(epsg= 4326)
        map_visualize(road_shape)
        

        return True

class MatchingPanos():
    def __init__(self, cache_folder="../cache", *args):
        self.memo = {}
        self.cache_folder = cache_folder
        if self.cache_folder is not None:
            self.load_memo()
        self.error_roads_lst = []
    
    def load_memo(self):
        if not os.path.exists(f'{self.cache_folder}/MatchingPanos_MEMO.pkl'):
            print(f"MatchingPanos loading {len(self.memo)} road from memo.pkl failed!")
            return False
        
        self.memo = pickle.load( open(f'{self.cache_folder}/MatchingPanos_MEMO.pkl', 'rb') )
        print(f"MatchingPanos loading {len(self.memo)} road from memo.pkl success!")
        return True
    
    def save_memo(self):
        pickle.dump(self.memo, open(f'{self.cache_folder}/MatchingPanos_MEMO.pkl', 'wb'))
        return True

    def add_lst(self, lst, df_edges, vis=False, debug=False):
        """[summary]

        Args:
            lst ([type]): [description]
            df_edges ([type]): [description]
            vis (bool, optional): [description]. Defaults to False.
            debug (bool, optional): [description]. save imgs to folder `.cache`.
        """
        for i in tqdm(lst, 'process road matching: '):
            self.add(i, vis, debug)

        return
    
    def add(self, i, vis=False, debug=False):
        if i in self.memo:
            return
        
        self.memo[i] = self.memo.get(i, {})
        df = get_and_filter_panos_by_osm_rid(i, vis=vis, debug=debug, outlier_filer=True)
        if df is None:
            self.error_roads_lst.append(i)
            self.memo[i]['df'] = None
            self.memo[i]['median'] = None
            
            return False
        
        self.memo[i]['df'] = df
        self.memo[i]['median'] = int(df.lane_num.median())
        
        return True
    
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

    def plot_matching(self, rid, *args, **kwargs):
        """plot the matching panos and show its lanenum in a color theme map

        Args:
            rid ([type]): [description]

        Returns:
            [type]: [description]
        """
        # matchingPanos.gdf_to_file_by_rid(name_to_id['科苑北路'], './')
        df = self.gdf_to_file_by_rid(rid)
        if df is None:
            print(f"plot rid matching error, for the geodataframe {rid} is None")
        
        df.loc[:, 'lane_num'] = df.loc[:, 'lane_num'].astype(str)
        _, ax = map_visualize(df, color='gray', scale=0.05, *args, **kwargs)
        df.plot(column='lane_num', ax=ax, legend=True)
        
        return df
    
    @property
    def size(self):
        return len(self.memo)
    
# sumo_net build-in functions 
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

def get_and_filter_panos_by_osm_rid(road_id = 243387686, offset=1, vis=False, debug=False, outlier_filer=True, mul_factor=2, verbose=True):
    """[summary]

    Args:
        road_id (int, optional): [description]. Defaults to 243387686.
        vis (bool, optional): [description]. Defaults to False.
        offset (int, optional): [the attribute `lane_num` is the real lane num or the real lane line num. If `lane_num` represent line num, then offset is 1. Other vise, the offset is 0 ]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    atts = ['index', 'RID', 'Name', 'geometry', 'lane_num', 'frechet_dis', 'angel', 'osm_road_id', 'osm_road_index', 'related_pos', 'link']
    try:
        if road_id > 0:
            matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False) 
            matching = matching[atts].merge(df_edges[['s', 'e']], left_on='osm_road_index', right_index=True)
            road_name = df_edges.query(f'rid=={road_id}').name.unique()[0]
        else:
            # FIXME -208128058 高新中三道, 街景仅遍历了一遍。。。。
            df_tmp = _get_revert_df_edges(road_id, df_edges)
            road_name = df_tmp.name.unique()[0]
            matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_tmp, False) 
            matching = matching[atts].merge(df_tmp[['s', 'e']], left_on='osm_road_index', right_index=True)
        if matching.shape[0] == 0:
            print( f"get_and_filter_panos_by_osm_rid {road_id}, no matching recods" )
            return None
    except:
        print( f"get_and_filter_panos_by_osm_rid {road_id}, no matching recods" )
        return None
    
    # filter outlier -> 计算路段的统计属性
    rids = matching.RID.unique().tolist()
    points = DB_panos.query( f"RID in {rids}" ).dropna()
    
    rid_order = CategoricalDtype(matching.RID, ordered=True)
    tmp = points.groupby('RID').apply( lambda x: _panos_filter(x) ).drop(columns='RID').reset_index()
    
    if outlier_filer and tmp.shape[0] != 0:
        if verbose: 
            origin_size = tmp.shape[0]
        _mean, _std = tmp.lane_num.mean(), tmp.lane_num.std()
        iterverl = (_mean-mul_factor*_std, _mean+mul_factor*_std)
        tmp.query( f" {iterverl[0]} < lane_num < {iterverl[1]}", inplace=True )
        if verbose: 
            print( f"size: {origin_size} -> {tmp.shape[0]}")
        
          
    if tmp.shape[0] == 0:
        print( f"get_and_filter_panos_by_osm_rid {road_id}, no matching recods after filter algorithm" )
        return None
     
    tmp.loc[:, 'RID'] = tmp['RID'].astype(rid_order)
    tmp.sort_values(by=['RID', 'Order'], inplace=True)
    tmp.reset_index(drop=True, inplace=True)

    if offset:
        tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'] - 1
        
    if debug or vis :
        fig, ax = map_visualize(tmp, scale=.1, color='gray', figsize=(15, 15))
        tmp.loc[:, 'lane_num_str'] = tmp.loc[:, 'lane_num'].astype(str)
        tmp.plot(ax=ax, column='lane_num_str', legend=True)
        
        _mean, _std = tmp.lane_num.mean(), tmp.lane_num.std()
        iterverl = (round(_mean-mul_factor*_std, 1), round(_mean+mul_factor*_std,1) )
        
        ax.set_title(f"{road_id}, {road_name}, mean {_mean:.1f}, std {_std:.1f}, {iterverl}", fontsize=18)
        if debug:
            try:
                fig.savefig(f'../cache/matching_records/{road_name}_{road_id}.jpg', dpi=300)
            except:
                print(road_name, road_id)
        plt.close()
        
    return tmp

def get_road_changed_section(rid, vis=True, dis_thres=20, mul_factor = 2):
    """获取变化的截面

    Args:
        rid ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    def _lane_seg_intervals(lane_num_dict):
        intervals = []
        for start, end, num in [(i, i+1, int(float(lane_num_dict[i]))) for i in lane_num_dict ]:
            merge_intervals(intervals, start, end, num)

        return intervals

    def _convert_interval_to_gdf(intervals, lines):
        change_pids = gpd.GeoDataFrame(intervals, columns=['pano_idx_0', 'pano_idx_1', 'lane_num'])

        change_pids.loc[:, 'pano_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].geometry )
        change_pids.loc[:, 'pano_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].geometry )
        change_pids.loc[:, 'pano_id_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].PID )
        change_pids.loc[:, 'pano_id_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].PID )
        change_pids.loc[:, 'rid0'] = change_pids.pano_0.apply( lambda x: lines.loc[lines.distance(x).argmin()].start )
        change_pids.loc[:, 'rid1'] = change_pids.pano_1.apply( lambda x: lines.loc[lines.distance(x).argmin()].end )
        change_pids.loc[:, 'length'] = change_pids.apply( lambda x: x.pano_0.distance(x.pano_1)*110*1000, axis=1 )

        return change_pids

    status = {0: 'sucess', 
              1: 'no matching record', 
              2: 'all the panos has the same lane_num',
              3: 'after filter, there is no availabel panos'}

    panos = OSM_MATCHING_MEMO[rid]['df']
    if panos is None or panos.shape[0] == 0:
        return None, status[1]
    segments = panos.query(" lane_num != @panos.lane_num.median() ")
    
    if segments.shape[0] == 0:
        return None, status[2]
    
    # 注意区间：左闭右开
    intervals = _lane_seg_intervals(segments['lane_num'].to_dict())

    pids = osm_net.get_pids_by_rid(rid, sumo_net)
    lines = gpd.GeoDataFrame([ {'index':i, 
                                'start': pids[i],
                                'end': pids[i+1],
                                'geometry': LineString( [osm_net.get_node_xy(pids[i]),
                                                         osm_net.get_node_xy(pids[i+1])] 
                                                       )
                                } for i in range(len(pids)-1) ],
                            ).set_crs(epsg=4326)

    # second layer for filter
    change_pids = _convert_interval_to_gdf(intervals, lines)
    change_pids.query("length != 0", inplace=True)
    _mean, _std = change_pids.lane_num.mean(), change_pids.lane_num.std()
    iterverl = (_mean-mul_factor*_std, _mean+mul_factor*_std)
    change_pids.query( f" {iterverl[0]} < lane_num < {iterverl[1]}", inplace=True )
        
        
    attrs = ['pano_idx_0', 'pano_idx_1', 'lane_num']
    keep = change_pids.query(f"length >= {dis_thres}")[attrs].values.tolist()
    if len(keep) < 1:
        return None, status[3]

    candidate = change_pids.query(f"length < {dis_thres}")[attrs].values.tolist()
    for i in candidate:
        keep = insert_intervals(keep, i)

    intervals = [i for i in keep if i not in candidate]
    change_pids = _convert_interval_to_gdf(intervals, lines)
    change_pids.loc[:, 'intervals'] = change_pids.apply( lambda x: [pids.index(x.rid0), pids.index(x.rid1)], axis=1 )
    change_pids.sort_values(by='intervals', inplace=True)
    
    return change_pids, status[0]

def osm_road_segments_intervals(x, pids):
    def helpler(x):
        if x in pids:
            return pids.index(x)
        
        if 'cluster' in x:
            id = max( [pids.index(int(i)) for i in x.split("_")[1:] if int(i) in pids ] )
        elif x.isdigit():
            id = pids.index(int(x))
        else:
            id = x
            
        return id

    return [helpler(x['from']), helpler(x['to'])]

def lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst, log=SUMO_LOG):
    lane_num_lst = [i[2] for i in new_intervals]
    for index, elem in enumerate(elem_lst):
        if elem is None: 
            continue
        
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
                    print(f'{id_lst[index]} removing origFrom')
                    
        if index != len(elem_lst) - 1:
            elem.set('to', str(pids[new_intervals[index][1]]))
            for record in elem.findall('param'):
                if 'origTo' == record.get('key'):
                    elem.remove(record)
                    print(f'{id_lst[index]} removing origTo')
        
        if index != 0:
            sumo_net.add_edge(elem)
        else:
            sumo_net.update_edge_df(elem)
        
        if elem.get('from') == elem.get('to'):
            sumo_net.remove_edge_by_rid(elem.get('id'))
        
        # `from` and `to` has the same id
        if  elem.get('to') in elem.get('from') or elem.get('from') in elem.get('to'):
            status = sumo_net.remove_edge_by_rid(elem.get('id'))
            SUMO_LOG.info(f"Remove_edge_by_rid\n\t{elem.get('id')}: {status}")
    
    return
        
def lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set, log=None, verbose=False):
    origin_start, origin_end = item.interval
    log_info = ["LANE_CHANGE_PROCESS"]

    s = origin_start if pd.isnull(item.origFrom) else pids.index(int(item.origFrom))
    e = origin_end   if pd.isnull(item.origTo)   else pids.index(int(item.origTo))
    new_intervals = [[s        , new_start, int(item.numLanes)], 
                     [new_start, new_end  , lane_num_new], 
                     [new_end  , e        , int(item.numLanes)]
                    ]
    new_intervals = [[i,j,k] for i,j,k in new_intervals if i != j]

    # check the distance of last interval 
    last_seg_dis = sys.maxsize
    if len(new_intervals) > 1:
        last_seg_dis = osm_net.cal_dis_two_point( pids[new_intervals[-1][0]], pids[new_intervals[-1][1]])
        if last_seg_dis < dis_thres:
            _, end, _ = new_intervals.pop()
            new_intervals[-1][1] = end
            
    last_seg_info = f'last dis {last_seg_dis:.0f}' if last_seg_dis < dis_thres  else ''
    log_info.append(f"\tsplit intervals {item.id}\n\t\torigin: [{origin_start}, {origin_end}], insert: [{new_start}, {new_end}], {last_seg_info} -> {str(new_intervals)}")

    assert origin_start <= origin_end, "check the pids interval"

    def _drop_pid_in_cluster(pids, verbose=True):
        # TODO 针对`231901941#9`线形优化，确定终点 [7782982560, 7782982556, 6444510067, 'cluster_7782982563_7782982564_7782982565']
        # osm_net.key_to_edge[231901941]
        pids_lst = pids.copy()
        cluster = []
        for i in pids_lst:
            if isinstance(i, str) and 'cluster' in i:
                cluster.append(i)

        cluster_pids = set()
        for item in cluster:
            for i in item.split('_')[1:]:
                i = int(i) if i.isdigit() else i
                cluster_pids.add(i)

        cluster_pids

        for pid in pids_lst:
            if pid in cluster_pids:
                pids_lst.remove(pid)

        if verbose:
            print(f"drop_pid_in_cluster: \n\t{pids} -> {pids_lst}")
        return pids_lst

    shape_lst, pids_lst = [], []
    for s, e, _ in new_intervals:
        sumo_net.check_node(pids[e], osm_net.key_to_node)
        sumo_net.check_node(pids[s], osm_net.key_to_node)
        
        # TODO 
        # pids_tmp = _drop_pid_in_cluster(pids[s:e+1])
        pids_tmp = [ p for p in  pids[s:e+1] if isinstance(p, int) or 'cluster' not in p]
        shape_tmp = " ".join( [",".join([ str(i) for i in osm_net.get_node_coords(p)]) for p in pids_tmp] )
        shape_lst.append(shape_tmp)
        pids_lst.append(pids_tmp)
    
    id_lst = []
    cur_order = item.order
    order_lst = [cur_order]
    for i in range(len(new_intervals)-1):
        while cur_order in order_set:
            cur_order += 1
        order_set.add(cur_order)
        order_lst.append(cur_order)
    # 208128050#8-AddedOffRampEdge
    postfix = "-"+item.id.split('-')[-1] if '-' in item.id[1:] else ''
    id_lst = [ f"{int(item.rid)}#{int(i)}{postfix}"  for i in order_lst ]
    # id_lst = [ f"{int(item.rid)}#{int(i)}"  for i in order_lst ]
    
    log_info.append(f"\n\tid: {id_lst}")
    log_info.append(f"\torder: {order_lst}")
    log_info.append(f"\tnew_intervals:{new_intervals}")
    log_info.append(f"\tpids_lst: {pids_lst}", )
    log_info.append(f"\tshape_lst: {shape_lst}", )
    
    if log is not None:
        log.info( "\n".join(log_info)+"\n" )
    if verbose:
        for i in log_info:
            print(i)
    
    origin_edge = sumo_net.get_edge_elem_by_id(item.id)
    elem_lst = [origin_edge] + [copy.deepcopy(origin_edge) for _ in range(len(new_intervals)-1)]
    
    # print(elem_lst)
    # for elem in elem_lst:
    #     print_elem(elem)
        
    lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst)

    if verbose:
        for _, elem in enumerate(elem_lst):
            print_elem(elem, '\t')
      
def modify_road_shape(rid, log=None, dis_thres=25, verbose=False):
    change_pids, status = get_road_changed_section(rid)
    attrs_show = ['id', 'from', 'to', 'numLanes', 'origFrom', 'origTo', 'order', 'interval']

    if change_pids is None:
        log.warning(f"Modify road shape [{rid}] failed, {status}\n")
        return
    
    road = sumo_net.get_edge_df_by_rid(rid)
    if road.shape[0] == 0:
        log.warning(f"Modify road shape [{rid}], not in the study area\n")
        return 
    
    pids = osm_net.get_pids_by_rid(rid, sumo_net, geo_plot=False)
    def _cal_and_sort_interval(road):
        road.loc[:, 'interval'] = road.apply(lambda x: osm_road_segments_intervals(x, pids), axis=1)
        road.sort_values('interval', inplace=True)
    
    _cal_and_sort_interval(road)    
    order_set = set( road.order.values )
    interval_min = road.interval.apply(lambda x: x[0]).min()
    interval_max = road.interval.apply(lambda x: x[1]).max()

    queue = deque( change_pids[['intervals', 'lane_num']].values.tolist() )
    if log is not None:
        log.notice(f"Modify road shape [{rid}], pids interval [{interval_min}, {interval_max}], processing\nqueue: {queue}\npids: {pids}\n\nsumo net dataframe:\n{road[attrs_show]}\n")
        
    while queue:
        if verbose: print("\n", queue)
        [new_start, new_end], lane_num_new = queue.popleft()
        # the case that the pids is not start or end with the pid specified in the osm file 
        new_start = interval_min if new_start < interval_min else new_start
        new_end   = interval_max if new_end   > interval_max else new_end
            
        if new_start == new_end:
            continue
        
        # SUMO_LOG.info(road[['id', 'from', 'to', 'numLanes', 'order', 'interval']])
        for index, item in road.iterrows():
            origin_start, origin_end = item.interval

            if origin_start >= new_end:
                if verbose: print( f"\n\tcase 1 origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                break
            elif origin_end <= new_start:
                if verbose: print( f"\n\tcase 2 origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                continue
            else:
                if new_start < origin_start and origin_start <= new_end <= origin_end:
                    queue.appendleft([[origin_start, new_end], lane_num_new ])
                    queue.appendleft([[new_start, origin_start], lane_num_new ])
                    if verbose: print( f"\n\tcase 3a origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                    break           
                
                if origin_start <= new_start <= origin_end and new_end > origin_end:
                    queue.appendleft([[origin_end, new_end], lane_num_new ])
                    queue.appendleft([[new_start, origin_end], lane_num_new ])
                    if verbose: print( f"\n\tcase 3b origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                    break
                
                lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set, log)
                if verbose: print( f"\n\tcase 3c origin_start: ", f"new [{new_start}, {new_end}], origin[{origin_start}, {origin_end}]", " -> ", queue )
                
                road = sumo_net.get_edge_df_by_rid(rid)
                # road.loc[:, 'interval'] = road.apply(lambda x: osm_road_segments_intervals(x, pids), axis=1)
                # road.sort_values('interval', inplace=True)
                _cal_and_sort_interval(road)    
    
                
                break
    
    return True

def _pre_process_fine_tune(name, osm_file, verbose=False, SUMO_HOME="/usr/share/sumo"):
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
        if verbose: print(i)
        if 'Writing network' in i:
            flag = True
    
    return flag

def _post_process_fine_tune(name, osm_file, verbose=False, SUMO_HOME="/usr/share/sumo"):
    """
    sumo releted process post fine tune
    """

    flag = False
    sumo_net.save()
    
    post_precess = f"cd ./{name}; ll; cp ../start_with_net.sh ./; sh start_with_net.sh "
    res = os.popen(post_precess).read()
    if verbose: print(res)
    
    if 'calling /usr/share/sumo/bin/duarouter' in res:
        flag = True
    
            
    return flag


#%%
# 初始化
rids_debug = [570468184, 633620767, 183402036, 107586308] # 深南大道及其辅道
file = open(f"../log/sumo-{datetime.datetime.now().strftime('%Y-%m-%d')}.log", 'w').close()

SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'


name_to_id = {'高新中四道': 529249851,
              '科技中二路': 208128052,
              '科苑北路': 231901939,
              '高新中二道': 208128050,
              '科技中三路': 208128048,
              '科技中一路': 278660698,
              '高新中一道': 778460597
              }

RID_set = [ 208128058, 529249851, -529249851, 208128051,208128050, 208128048,489105647, 231901941,778460597] # -208128058

if matchingPanos is None:
    matchingPanos = MatchingPanos()
    # matchingPanos = MatchingPanos(None)
    # matchingPanos.add_lst(RID_set, df_edges, debug=True)
    # matchingPanos.save_memo()

osm_net = OSM_Net(file='./osm_bbox.osm.bak.xml', save_fn='./osm_bbox.osm.xml', logger=SUMO_LOG)

rids = []
road_levles = ['trunk'] # 'primary', 'secondary', 
for level in road_levles:
    rids += osm_net.get_roads_by_road_level(level)

matchingPanos.add_lst(rids if rids_debug is None else rids_debug, df_edges, debug=True)
OSM_MATCHING_MEMO = matchingPanos.memo

if rids_debug is None:
    osm_net.rough_tune(rids+RID_set, OSM_MATCHING_MEMO, save=True)
else:
    osm_net.rough_tune(rids_debug if isinstance(rids_debug, list) else [rids_debug], OSM_MATCHING_MEMO, save=True)
    
assert _pre_process_fine_tune(name, osm_file, False), 'check `_pre_process_fine_tune` functions'

sumo_net = Sumo_Net('osm', logger=SUMO_LOG)
osm_net.add_sumo_net_node_to_osm(sumo_net)
osm_net.add_coords_to_node(OSM_CRS)

"""微调"""
if rids_debug is None:
    fine_tune_road_set = [ 208128050, 208128058, 208128051, 529249851, 208128048, 489105647, 778460597,  231901941]
    for rid in rids + fine_tune_road_set:
        try:
            modify_road_shape(rid, SUMO_LOG)
        except:
            print(f"rid: {rid} error!" )
            SUMO_LOG.error(f"rid: {rid} error!" )
            break
else:
    if isinstance(rids_debug, list):
        [ modify_road_shape(i, SUMO_LOG) for i in rids_debug ]
    else:
        modify_road_shape(rids_debug, SUMO_LOG)

assert _post_process_fine_tune(name, osm_file, False), 'check `_post_process_fine_tune` functions'


        
# %%
"""
    DONE:
    572963461 -> 匹配的elem_lst is [None] 
    107586308
    TODO
    911272994 # if panos is None or panos.shape[0] == 0: 
    25529503 # 2号道路，神奇失踪正在溯源
"""
rid = 572963460
sumo_net.plot_edge(rid)
sumo_net.get_edge_df_by_rid(rid)

osm_net.get_pids_by_rid(rid, sumo_net)

modify_road_shape(rid, SUMO_LOG)

# panos matching
panos = get_and_filter_panos_by_osm_rid( rid, vis=True, debug=False )




# %%
# ! 针对panos匹配的情况进行异常值处理，从图的连通性角度出发
panos



# %%

# FIXME rid = 107586308， 存在和 `220885829`合并的情况

# bug 线段延申的情况
# rid = 231405165
# modify_road_shape(rid, SUMO_LOG)

# bug 没有预测错误的项目
# rid = 636237018
# modify_road_shape(rid, SUMO_LOG)


# 45569111 生成错误, 因为线段被裁剪头部
# rid = 45569111
# modify_road_shape(rid, SUMO_LOG)

# 修改匹配算法，剪枝
rid = 107586308

#%%
#! 获取sumo_net的 拓扑 关系
rid = '107586308#7-AddedOnRampEdge'

df_edge = sumo_net.edge

index = df_edge.query(f"id=='{rid}' ").index[0]

road = df_edge.loc[index]
end_point = road['to'] 
rid = road['rid']

# %%
pre_ramps = df_edge[ (df_edge['to'] == road['from']) & (df_edge['rid'] != road['rid']) ]
pre_ramps_numLanes = pre_ramps.numLanes.astype(int).sum()

nxt_ramps = df_edge[ (df_edge['from'] == road['to']) & (df_edge['rid'] != road['rid']) ]
nxt_ramps_numLanes = nxt_ramps.numLanes.astype(int).sum()

lane_num = int(road['numLanes']) - max(pre_ramps_numLanes, nxt_ramps_numLanes)

sumo_net.update_edge_elem_lane_num(rid, lane_num)

# %%

# rid = '107586308#7'

def update_laneNum_for_AddRampEdge(rid, verbose=True):
    df_edge = sumo_net.edge

    index = df_edge.query(f"id=='{rid}' ").index[0]
    road = df_edge.loc[index]
    # rid = road['rid']

    if verbose:
        print("\n\n", road['id'], road['numLanes'])

    pre_ramps = df_edge[ (df_edge['to'] == road['from']) & (df_edge['rid'] != road['rid']) & (df_edge['type'] != road['type']) ]
    pre_ramps_numLanes = pre_ramps.numLanes.astype(int).sum()

    nxt_ramps = df_edge[ (df_edge['from'] == road['to']) & (df_edge['rid'] != road['rid']) & (df_edge['type'] != road['type']) ]
    nxt_ramps_numLanes = nxt_ramps.numLanes.astype(int).sum()

    tmp_lane = int(road['numLanes']) - max(pre_ramps_numLanes, nxt_ramps_numLanes)
    lane_num = tmp_lane if tmp_lane > 0 else int(road['numLanes'])

    status = sumo_net.update_edge_elem_lane_num(rid, lane_num)

    if status is False:
        print(f"update_laneNum_for_AddRampEdge {rid} failed")
    else:
        print(f"update_laneNum_for_AddRampEdge {rid} success")
    

    return 
    

# update_laneNum_for_AddRampEdge(rid='107586308#7-AddedOnRampEdge')    


# %%
add_ramps_edges = sumo_net.edge[sumo_net.edge['id'].str.contains('107586308') & sumo_net.edge['id'].str.contains('RampEdge')]['id'].values

# %%
for edge in add_ramps_edges:
    update_laneNum_for_AddRampEdge(edge)

assert _post_process_fine_tune(name, osm_file, False), 'check `_post_process_fine_tune` functions'


# %%
