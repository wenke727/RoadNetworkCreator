
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer


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

    def get_node(self, pid:str):
        if str(pid) not in self.key_to_node:
            return None
        return self.key_to_node[str(pid)]
    
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

