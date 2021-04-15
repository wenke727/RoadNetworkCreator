#%%
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
from road_network import OSM_road_network
from road_matching import *
# from road_matching import _matching_panos_path_to_network, get_panos_of_road_and_indentify_lane_type_by_id, df_edges, DB_panos, DB_roads

from utils.geo_plot_helper import map_visualize

 # TODO: reversible, alternating: https://wiki.openstreetmap.org/wiki/Tag:oneway%3Dreversible
ONEWAY_DICT = {'yes': True, 
                '-1': True, 
                '1': True, 
                'reversible': True,
                'no': False,
                '0': False
    }

OSM_CRS = None

#%%
# move to osm helper
def osm_get(prefix, bbox):
    # bounding box to  in geo coordinates west,south,east,north
    bbox   = '113.92348,22.57034,113.94372,22.5855'
    prefix = "osm" if prefix is None else prefix
    cmd = f"python /usr/share/sumo/tools/osmGet.py -b {bbox} -p {prefix}"
    os.popen( cmd )
    
    return f"osm_{prefix}_bbox.osm.xml" if prefix != 'osm' else "osm_bbox.osm.xml"


def osm_parser(root):
    wayHash, nodeHash = {}, {}
    for element in root.findall('way'):
        id = int(element.get('id'))
        if id not in wayHash:
            wayHash[id] = {}
            wayHash[id]['elem'] = element
            wayHash[id]['points'] = [ int(i.get('ref')) for i in element.findall('nd')]
            
            for i in element.findall('tag'):
                wayHash[id][i.get('k')] = i.get('v')
            
    for node in root.findall('node'):
        id = int(node.get('id'))
        if id in nodeHash: continue

        info = {x.get('k'):x.get('v') for x in node.getchildren()} if node.getchildren() else {}
        info['pid'] = id
        info['xy'] = (float(node.get('lon')), float(node.get('lat')))
        info['geometry'] = Point( *info['xy'] )
        nodeHash[id] = info

    return wayHash, nodeHash


def add_coords_to_osm_node_hash(osm_nodeHash):
    """add projection coordinations of each node to osm node hash

    Args:
        osm_nodeHash (dict): osm node converted from osm xml file 

    Returns:
        [type]: [description]
    """
    df = gpd.GeoDataFrame(osm_nodeHash).T
    df = df.set_crs(epsg=4326).to_crs(epsg=OSM_CRS)
    df.loc[:, 'coords'] = df.geometry.apply(lambda x: [ round(i,2) for i in x.coords[0]])

    for i in osm_nodeHash.keys():
        osm_nodeHash[i]['coords'] = df.loc[i, 'coords']
        
    return osm_nodeHash


def tranfer_to_sumo(project_name='nanshan', osm_file="./osm_bbox.osm.xml"):
    res = os.popen(f" mkdir {project_name}; cp {osm_file} ./{project_name}/osm_bbox.osm.xml; \
                      cp ./start.sh ./{project_name} ; cd ./{project_name};sh start.sh").read()

    return True


# move to xml helper
def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    return


def update_element_attrib(item, key, val, log=False):
    def _print():
        print({tag.get('k'): tag.get('v') for tag in item.findall('tag')})
    
    tags = [tag.get('k') for tag in item.findall('tag')]
    if log: _print()
    
    # 判断是否为双向道路，若是则需要将val/2  # <tag k="oneway" v="yes" />
    if key =='lanes':
        oneway_att = False
        if 'oneway' in tags:
            element = item.findall('tag')[tags.index('oneway')]
            oneway_att = ONEWAY_DICT[element.get('v')]
        
        if not oneway_att and isinstance(val, int):
            val = val *2

    if key in tags:
        element = item.findall('tag')[tags.index(key)]
        element.set('v', str(val))
        if log: _print()
        return True

    lanes_info = ET.Element('tag', {'k': key, 'v': str(val)})
    item.append(lanes_info)
    if log: _print()

    return True


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


# inside
def merge_index_to_intervals(nums):
    '''
    test example：
    nums = [32, 33, 34, 35] + [1,3,4,5] + [37]
    '''
    def _merge_to(intervals, start, end):
        # if start is None or start == end:
        if start is None: return
        
        if not intervals:
            intervals.append([start, end])
            return
        
        _, prev_end = intervals[-1]
        if  prev_end == start:
            intervals[-1][1] = end
            return
        
        intervals.append([start, end])

    nums.sort()
    tmp = [ [nums[i], nums[i+1]] if nums[i] - nums[i+1]==-1 
            else [nums[i], nums[i]] for i in range(len(nums)-1) ] + [[nums[-1], nums[-1]]]

    intervals = []
    for start, end in tmp:
        _merge_to(intervals, start, end)
            

    return intervals


def parser_sumo_node_edge(name):
    """parse the sumo node and edge files created by netconvet from OSM file

    Args:
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    global OSM_CRS

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
    
    net = {}
    net['edge_tree'] = ET.parse(f'./{name}/{name}.edg.xml')
    net['edge_root'] = net['edge_tree'].getroot()
    
    net['node_tree'] = ET.parse(f'./{name}/{name}.nod.xml')
    net['node_root'] = net['node_tree'].getroot()
    
    net['key_to_edge'], net['key_to_node'] = {}, {}
    edges, nodes = [], []
    OSM_CRS = CRS(net['node_root'].findall('location')[0].get('projParameter')).to_epsg()
    
    for item in net['edge_tree'].findall('edge'):
        net['key_to_edge'][item.get('id')] = item
        info = {key: item.get(key) for key in  item.keys()}
        for p in item.findall('param'):
            info[p.get('key')] = p.get('value')
        edges.append(info)

    edges = pd.DataFrame(edges)
    edges.loc[:, 'rid'] = edges.id.apply( _id_parser )
    edges.loc[:, 'order'] = edges.id.apply( _order_parser )

    for item in net['node_root'].findall('node'):
        net['key_to_node'][item.get('id')] = item
        nodes.append({key: item.get(key) for key in item.keys()})

    nodes = gpd.GeoDataFrame(nodes)
    nodes.loc[:, 'geometry'] = nodes.apply( lambda x: Point( float(x.x), float(x.y) ) ,axis=1 )
    nodes.set_crs(epsg=OSM_CRS, inplace=True)

    net['edge'], net['node'] = edges, nodes

    return net


def df_coord_transfor(gdf):
    # 坐标转换算法
    # <location netOffset="-799385.77,-2493897.75" convBoundary="0.00,0.00,18009.61,5593.04" origBoundary="113.832744,22.506539,114.086290,22.692155" projParameter="+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>

    crs = pyproj.CRS("+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    to_crs = crs.to_epsg()

    gdf.to_crs(epsg=to_crs, inplace = True)

    nodes.loc[:, "x_"] = gdf.geometry.x - 799385.77
    nodes.loc[:, "y_"] = gdf.geometry.y - 2493897.75

    return gdf


# temp 
def osm_edge_shape_to_linestring():
    tmp = edges.query( 'rid== 208128052 ' )

    tmp = gpd.GeoDataFrame(tmp, geometry=  tmp['shape'].apply( lambda x: LineString([ np.array(i.split(",")).astype(np.float)  for i in x.split(' ')])) )
    tmp.plot()



#%%
def matching_and_predicting_panos(RID_set):
    def _get_revert_df_edges(road_id, vis=False):
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
            df_tmp = _get_revert_df_edges(road_id)
            matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_tmp, False) 
            matching = matching[atts].merge(df_tmp[['s', 'e']], left_on='osm_road_index', right_index=True)
            
        # 过滤异常值 -> 计算路段的统计属性
        rids = matching.RID.unique().tolist()
        points = DB_panos.query( f"RID in {rids}" ).dropna()

        def _panos_filter(panos):
            if panos.shape[0] == 2 and panos.lane_num.nunique() == 1:
                return panos

            remove_pano_num = 1
            median = int(np.median(panos.lane_num))
            remain_ponas_index = np.sort(panos.Order.unique())[remove_pano_num: -remove_pano_num]

            tmp = panos[['Order','lane_num']]
            tmp.loc[:, 'prev'] = panos.lane_num.shift(-1) == panos.lane_num
            tmp.loc[:, 'nxt'] = panos.lane_num.shift(1) == panos.lane_num
            not_continuous = tmp[(tmp.prev | tmp.nxt) == False].Order.values.tolist()

            panos.query( f" Order not in {not_continuous} and \
                            Order in @remain_ponas_index and \
                            abs(lane_num - @median) < 2", 
                            inplace=True 
                        )
            
            return panos

        rid_order = CategoricalDtype(matching.RID, ordered=True)
        tmp = points.groupby('RID').apply( lambda x: _panos_filter(x) ).drop(columns='RID').reset_index()
        tmp.loc[:, 'RID'] = tmp['RID'].astype(rid_order)
        tmp.sort_values(by=['RID', 'Order'], inplace=True)
        tmp.reset_index(drop=True, inplace=True)

        if offset:
            tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'] - 1
            
        if vis:
            fig, ax = map_visualize(tmp, scale=.1, color='gray', figsize=(15, 15))
            
            tmp.loc[:, 'lane_num_str'] = tmp.loc[:, 'lane_num'].astype(str)
            tmp.plot(ax=ax, column='lane_num_str', legend=True)

            # fig, ax = map_visualize(tmp, scale=.1, color='gray', figsize=(15, 15))
            # tmp.loc[:, 'index'] = tmp.index
            # tmp.plot(ax=ax, column='index', legend=True)

        return tmp

    OSM_MATCHING_MEMO = {}
    for i in RID_set:
        OSM_MATCHING_MEMO[i] = OSM_MATCHING_MEMO.get(i, {})
        df = get_and_filter_panos_by_osm_rid(i, vis)
        OSM_MATCHING_MEMO[i]['df'] = df
        OSM_MATCHING_MEMO[i]['median'] = int(df.lane_num.median())

    return OSM_MATCHING_MEMO

# road_id = 529070115 # 打石一路
# road_id = 243387686 # 创科路
road_id = 208128052 # 科技中二路

RID_set = [road_id, -road_id]
OSM_MATCHING_MEMO = matching_and_predicting_panos(RID_set, vis=True)


#%% initial
SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'

tree = ET.parse('./osm_bbox.osm.bak.xml')
root = tree.getroot()
# 为osm里边的对象
osm_wayHash, osm_nodeHash = osm_parser(root)

"""粗调"""
rid = RID_set[0]
lane = OSM_MATCHING_MEMO[rid]['median']
item = osm_wayHash[rid]
update_element_attrib(item['elem'], 'lanes', int(lane))


# internal functions
def _pre_process_fine_tune():
    pre_process = f' rm -r ./{name}; mkdir {name}; cp {osm_file} ./{name}/{osm_file}; cd ./{name}'

    cmd = f"""
        {SUMO_HOME}/bin/netconvert  -t {SUMO_HOME}/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files {osm_file} --keep-edges.by-vclass passenger --offset.disable-normalization true -o {name}.net.xml
    """

    # create node, edge files
    cmd_tranfer0 = f"""{SUMO_HOME}/bin/netconvert --sumo-net-file {name}.net.xml --plain-output-prefix {name}; """
    for i in os.popen(' '.join( [pre_process, cmd, cmd_tranfer0] )).read().split('\n'): print(i)

def _post_process_fine_tune():
    cmd_bck_net = f"""cd ./{name}; mv {name}.net.xml {name}_old.net.xml;
        {SUMO_HOME}/bin/netconvert --node-files {name}.nod.xml --edge-files {name}.edg.xml -t {name}.typ.xml  --precision 2 --precision.geo 6 --offset.disable-normalization true -o {name}.net.xml
    """ 
    post_precess = " cp ../start_with_net.sh ./; sh start_with_net.sh "
    for i in os.popen(' '.join( [cmd_bck_net, post_precess ] )).read().split('\n'): print(i)
    return

def _get_chang_road_section(rid, vis=True):
    """获取变化的截面

    Args:
        rid ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # TODO add `lane_num`
    panos = OSM_MATCHING_MEMO[rid]['df']
    segments = panos.query(" lane_num != @ panos.lane_num.median() ")
    intervals = np.array(merge_index_to_intervals( segments.index.values ))

    # gpd.GeoDataFrame({i: osm_nodeHash[i] for i in osm_wayHash[rid]['points']}).T.reset_index()
    # gpd.GeoDataFrame([ osm_nodeHash[i] for i in osm_wayHash[rid]['points']] )
    pids = osm_wayHash[rid]['points']
    lines = gpd.GeoDataFrame([ {'index':i, 
                                'start': osm_nodeHash[pids[i]]['pid'],
                                'end': osm_nodeHash[pids[i+1]]['pid'],
                                'geometry': LineString( [osm_nodeHash[pids[i]]['xy'],
                                                        osm_nodeHash[pids[i+1]]['xy']] )} 
                                for i in range(len(pids)-1) ],
                            ).set_crs(epsg=4326)

    change_pids = gpd.GeoDataFrame(intervals, columns=['s', 'e'])
    change_pids.loc[:, 'pano0']  = change_pids.s.apply( lambda x: panos.loc[x].geometry )
    change_pids.loc[:, 'pano1']  = change_pids.e.apply( lambda x: panos.loc[x].geometry )
    change_pids.loc[:, 'rid0'] = change_pids.pano0.apply( lambda x: lines.loc[lines.distance(x).argmin()].start )
    change_pids.loc[:, 'rid1'] = change_pids.pano1.apply( lambda x: lines.loc[lines.distance(x).argmin()].end )

    change_pids.loc[:, 'intervals'] = change_pids.apply( lambda x:  [osm_wayHash[rid]['points'].index(x.rid0), osm_wayHash[rid]['points'].index(x.rid1)]  , axis=1 )
    
    return change_pids

def osm_road_segments_intervals(x, plst = osm_wayHash[rid]['points']):
    def helpler(x):
        if 'cluster' in x:
            id = max( [  plst.index(int(i)) for i in x.split("_")[1:] ] )
        else:
            id = plst.index(int(x))
        
        return id

    return [helpler(x['from']), helpler(x['to'])]



indent(root)
tree.write('osm_bbox.osm.xml', encoding='utf-8')
_pre_process_fine_tune()
# tranfer_to_sumo()


#%%

sumo_net = parser_sumo_node_edge(name)
add_coords_to_osm_node_hash(osm_nodeHash)

# sumo_net['edge'].query(f"rid=={-rid}")
# sumo_net['edge'].order.value_counts()

"""微调"""
osm_wayHash[rid]
change_pids = _get_chang_road_section(rid)
# TODO add lane_num
change_pids.loc[:, 'lane_num'] = 3


road = sumo_net['edge'].query(f"rid=={rid}").sort_values('order')
road.loc[:, 'interval'] = road.apply(osm_road_segments_intervals, axis=1)
# road = road[['id', 'from', 'to', 'numLanes', 'speed', 'shape', 'rid', 'order', 'interval']]
road

# 生成
# _post_process_fine_tune()


# %%
# @param
dis_thres = 25


road = sumo_net['edge'].query(f"rid=={rid}").sort_values('order')
road.loc[:, 'interval'] = road.apply(osm_road_segments_intervals, axis=1)

points = osm_wayHash[rid]['points']
order_set = set( road.order.values )

queue = change_pids[['intervals', 'lane_num']].values.tolist()
[new_start, new_end], num = queue[0]

def _check_pano_in_SUMO_node(pid, log=True):
    """check the pano in SUMO nod.xml or not, if not exist then insert the record

    Args:
        pid (long): the pid of node in OSM
    """
    # notice the pid in sumo_net['key_to_node'] is string instead of long
    if str(pid) in sumo_net['key_to_node']:
        return False
    
    x, y = osm_nodeHash[pid]['coords']
    info = {"id": str(pid), 'x': str(x), 'y': str(y)}
    node = ET.Element('node', info)
    sumo_net['node_root'].append(node)
    sumo_net['key_to_node'][str(pid)] = node
    
    if log: print(f'insert node into SUMO node xml file:\n\t{info}\n')
    
    return True

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

def print_elem(elem, indent='', print_child=True):
    print(indent, {i: elem.get(i) for i in elem.keys()}) 
    if not print_child:
        return
    
    for i in elem.getchildren():
        print_elem(i, indent+'\t', print_child)    
    return 

for index, item in road.iterrows():
    origin_start, origin_end = item.interval
    origin_edge = sumo_net['key_to_edge'][item.id]
    
    # print(item.id)
    if origin_start > new_end:
        # print(f'\tbreak: {item.id}')
        break
    elif origin_end < new_start:
        continue
    else:
        print(f"split intervals {item.id}, origin: [{origin_start}, {origin_end}], insert: [{new_start}, {new_end}]")

        # TODO split functions
        # if not pd.isnull(item.origTo): origin_end = points.index(int(item.origTo))
        # if not pd.isnull(item.origFrom): origin_start = points.index(int(item.origFrom))
        new_intervals = [[origin_start if pd.isnull(item.origFrom) else points.index(int(item.origFrom)), new_start], 
                         [new_start, new_end], 
                         [new_end, origin_end if pd.isnull(item.origTo) else points.index(int(item.origTo))]
                        ]
        # check the distance of last interval 
        last_seg_dis = cal_dis_two_point( points[new_end], points[new_intervals[-1][1]])
        if last_seg_dis < dis_thres:
            _, end = new_intervals.pop()
            new_intervals[-1][1] = end
        print(f"new_intervals: {new_intervals}, the last segment dis is {last_seg_dis}")
        
        shape_lst = []
        shape = item.get('shape')
        if False:
            for s, e in new_intervals[1:]:
                _check_pano_in_SUMO_node(points[s])
                coord_str = ",".join([ str(i) for i in osm_nodeHash[points[s]]['coords']])
                if coord_str in shape:
                    a, shape = shape.split(coord_str)
                    a += coord_str
                    shape = coord_str + shape
                    shape_lst.append(a)
            shape_lst.append(shape)
        for s, e in new_intervals:
            _check_pano_in_SUMO_node(points[s])
            _check_pano_in_SUMO_node(points[e])
            tmp =  " ".join( [",".join([ str(i) for i in osm_nodeHash[p]['coords']]) for p in points[s:e+1]] )
            shape_lst.append(tmp)
        
        id_lst = []
        cur_order = item.order
        order_lst = [cur_order]
        for i in range(len(new_intervals)-1):
            while cur_order in order_set:
                cur_order += 1
            order_set.add(cur_order)
            order_lst.append(cur_order)
        id_lst = [ f"{item.rid}#{i}"  for i in order_lst ]
        
        print(f"id:\n\t{id_lst}")
        
        print(f"order:\n\t{order_lst}")

        print(f"new_intervals:\n\t{new_intervals}")

        print(f"\nshape_lst:\n\t {shape_lst}", )
        
        elem_lst = [origin_edge, copy.deepcopy(origin_edge)]
        for index, elem in enumerate(elem_lst):
            print(index)
            
        break
    
            

# print_elem(origin_edge)   

# print_elem(sumo_net['key_to_edge']['208128052#8'])

# sumo_net related to edge: edge_root, key_to_edge, edge


# indent(sumo_net['node_tree'].getroot())
# sumo_net['node_tree'].write('./osm/osm_new.nod.xml', encoding='utf-8')
# cal_dis_two_point( points[19], points[17] )




# %%
# ! modified attributes in node

lane_num_lst = [2, 3]
elem_lst = [copy.deepcopy(origin_edge), copy.deepcopy(origin_edge)]
# elem_lst = [origin_edge, copy.deepcopy(origin_edge)]
for index, elem in enumerate(elem_lst):
    elem.set('id', id_lst[index])
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
        elem.set('from', str(points[new_intervals[index][0]]))
        for record in elem.findall('param'):
            if 'origFrom' == record.get('key'):
                elem.remove(record)
                print('removing origFrom')
                
    if index != len(elem_lst) - 1:
        elem.set('to', str(points[new_intervals[index][1]]))
        for record in elem.findall('param'):
            if 'origTo' == record.get('key'):
                elem.remove(record)
                print('removing origTo')
    
    print_elem(elem)
    
    sumo_net['edge_root'].append(elem)

print('\n\n')
print_elem(origin_edge)   

sumo_net['edge_root'].remove(origin_edge)

# %%
indent(sumo_net['node_tree'].getroot())
indent(sumo_net['edge_tree'].getroot())
sumo_net['node_tree'].write('./osm/osm.nod.xml', encoding='utf-8')
sumo_net['edge_tree'].write('./osm/osm.edg.xml', encoding='utf-8')
# %%
_post_process_fine_tune()



# %%
# ! revert direction edge
matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, True)



# %%

_get_revert_df_edges(road_id, False)
# %%
