import os
import sys
import copy
import pyproj
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import geopandas as gpd
import xml.etree.ElementTree as ET
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString

sys.path.append("../src")


def osm_get(prefix, bbox):
    # bounding box to  in geo coordinates west,south,east,north
    bbox   = '113.92348,22.57034,113.94372,22.5855'
    prefix = "osm" if prefix is None else prefix
    cmd = f"python /usr/share/sumo/tools/osmGet.py -b {bbox} -p {prefix}"
    os.popen( cmd )
    
    return f"osm_{prefix}_bbox.osm.xml" if prefix != 'osm' else "osm_bbox.osm.xml"


def proj_trans(x, y, in_sys=4326, out_sys=32649, offset=(-799385.77,-2493897.75), precision=2 ):
    """proj trans

    Args:
        x ([type]): [description]
        y ([type]): [description]
        in_sys (int, optional): [description]. Defaults to 4326.
        out_sys (int, optional): [description]. Defaults to 32649.
        offset (tuple, optional): [description]. Defaults to (-799385.77,-2493897.75).
        precision (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # assert not isinstance(coord, tuple) and len(coord) != 2, "check coord"
    
    # always_xy (bool, optional) – If true, the transform method will accept as input and return as output coordinates using the traditional GIS order, that is longitude, latitude for geographic CRS and easting, northing for most projected CRS. 
    coord_transfer = Transformer.from_crs( CRS(f"EPSG:{in_sys}"), CRS(f"EPSG:{out_sys}"), always_xy=True )
    x, y = coord_transfer.transform(x, y)
    x += offset[0]
    y += offset[1]
    
    return round(x, precision), round(y, precision)




# visulize
# df = gpd.GeoDataFrame( [ nodeHash[i] for i in pids], index=pids )
# plt,ax = map_visualize(df, figsize=(15, 15))
# df.plot(column = 'highway', legend=True, ax=ax)

# proj_trans(113.954112, 22.544043)

# 科技中二路北行第一段
# <node id="8349563238" lat="22.5442170" lon="113.9336179">
pid = 8349563238
proj_trans(113.9336179, 22.5442170) # (2349.5, 2131.77)

# 从osm中提取线段信息
df_way = []
for index, item in wayHash.items():
    nodes = [ int(i.get('ref')) for i in  item.findall('nd')]
    df_way.append( {'id':int(item.get('id')), 'ps':nodes} )

df_way = gpd.GeoDataFrame(df_way)
df_way.loc[:, 'geometry'] = df_way.ps.apply( lambda x: LineString( [ nodeHash[i]['geometry'].coords[0] for i in x ] )) 

df_way.loc[:,'rid'] = df_way.id
df_way.loc[:,'name'] = df_way.id

df_way.set_index('id', inplace=True)


df_way.loc[208128052].geometry


edges.merge( df_edges, left_on =['from', 'to'], right_on=['s', 'e'] )
road = df_way.loc[208128052]
road_candidates, rid = _matching_panos_path_to_network( road, vis=False, vis_step=False )


#%%
# def parser_sumo_node_edge(name):

net = {}
tree = ET.parse('./osm/osm.nod.xml')
root = tree.getroot()
net['node_root'] = root

nodes = {}
for item in root.findall('node'):
    id = item.get('id')
    if id not in nodes:
        nodes[id] = { i: item.get(i) for i in item.keys()}
net['nodes'] = pd.DataFrame(nodes).T

tree = ET.parse


# %%

# developing
def get_points( rid, nodeHash):
    pids = [ x.get('ref') for x in  wayHash[str(rid)].findall('nd')]
    df = gpd.GeoDataFrame({ 'id': pids, 'geometry': [ nodeHash[x] for x in pids]  })
    df.reset_index()
    
    return df, pids


def split_road_and_update_lanes_val(rid, nodeHash):
    # 实现路口拓宽功能
    # @parma
    splits=['6444510067', '7782982564']
    lane_values = [3,4,3]

    df, pids = get_points(rid, nodeHash)
    
    splits_index = [pids.index(i) for i in splits] + [len(pids)]
    intervals, prev = [], 0
    for i in splits_index:
        intervals.append( pids[prev: i+1] )
        prev = i

    for index, inter in enumerate(intervals):
        new_way = copy.deepcopy(wayHash[str(rid)])
        for node in new_way.findall('nd'):
            if node.get('ref') not in inter:
                new_way.remove( node )
        
        # TODO set a unique id 
        new_way.set('id', new_way.get('id') + str(index))
        update_element_attrib(new_way, 'lanes', lane_values[index])
        
        root.append(new_way)

    root.remove( wayHash[str(rid)] )


def __add_revert_road():
    rid = 208128052 # 科技中二路
    # origin
    update_element_attrib(wayHash[str(rid)], 'oneway', 'yes')
    # revert edge
    way =  copy.deepcopy(wayHash[str(rid)])
    nodes = way.findall('nd')[::-1]

    for node in nodes:
        way.remove(node)

    df, pids = get_points(rid, nodeHash)

    line_offset = LineString( [ nodeHash[p].geometry.coords[:][0] for p in pids] ).\
        parallel_offset(3/110/1000, 'Left', join_style=2, mitre_limit=5)

    id = -int(pids[0])
    for node in line_offset.coords[::-1]:
        node_element = ET.Element('node', { 'id': str(id),'lat': f"{node[1]}", 'lon':f"{node[0]}"})
        root.append(node_element)
        
        nd_elem = ET.Element('nd', { 'ref': str(id)})
        way.append( nd_elem )

        id -= 1

    # update_element_attrib(way, 'oneway', 'yes')
    way.set('id', '-'+way.get('id') )
    root.append(way)


    root = indent(root)
    tree.write('osm_bbox.osm.xml', encoding='utf-8')

    tranfer_to_sumo()

    # TODO 整体平移, 增加节点，

    df, pids = get_points(rid, nodeHash)

    line_offset = LineString( [ nodeHash[p].geometry.coords[:][0] for p in pids] ).\
        parallel_offset(20/110/1000, 'Left', join_style=2, mitre_limit=5)

    start_id = -int(pids[0])

    for node in line_offset.coords[::-1]:
        node_element = ET.Element('node', { 'id': str(id),'lat': str(y), 'lon': str(x)})
        root.append(node_element)
        id -= 1
        
        nd_elem = ET.Element('nd', { 'ref': str(id)})
        way.append( nd_elem )

