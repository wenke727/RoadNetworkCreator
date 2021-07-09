
import sys
import copy
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from collections import deque
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer
from xml_helper import indent, update_element_attrib, print_elem

sys.path.append("../src")
from utils.geo_plot_helper import map_visualize

from utils.spatialAnalysis import relation_bet_point_and_line


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

    def save(self, name):
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

    def get_rids_by_road_level(self, level):
        if level not in self.road_level_dict:
            print(f'There is no {level} roads in the dataset')
            return []

        return list(self.road_level_dict[level])


    def get_rids_by_road_levels(self, road_types_lst:list):
        """Get rids by road type: 'primary', 'secondary'

        Args:
            road_types_lst (list): [description]
        """
        rids = [] 
        for level in road_types_lst:
            rids += self.get_rids_by_road_level(level)
        
        return rids
    
        
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
                flag = relation_bet_point_and_line(self.get_node_coords(node), 
                                                   self.get_node_coords(pids[idx]) + self.get_node_coords(pids[idx+1]))
                if 0 <= flag <= 1:
                    break
            
            if flag <= 1:
                pids.insert(idx+1, node)
            else:
                pids += [node]

        if geo_plot:
            lst = pids
            gdf_nodes = gpd.GeoDataFrame(geometry=[self.key_to_node[node]['geometry'] for node in lst])
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

