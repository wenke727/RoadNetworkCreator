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

from osm_helper import osm_get, osm_parser, add_coords_to_osm_node_hash, tranfer_to_sumo
from xml_helper import indent, update_element_attrib, print_elem

OSM_CRS = None

#%%
# function 
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
    # TODO 优化过滤算法， 结合后边计算间隔距离，若小于20m则忽略了
    if panos.shape[0] == 2 and panos.lane_num.nunique() == 1:
        return panos

    remove_pano_num = 1
    median = int(np.median(panos.lane_num))
    remain_ponas_index = np.sort(panos.Order.unique())[remove_pano_num: -remove_pano_num]

    # panos = DB_panos.query("RID=='affdd8-a23c-a68e-bddb-2f284e'")
    tmp = panos[['Order','lane_num']]
    # tmp.loc[:, 'prev'] = panos.lane_num.shift(-1) == panos.lane_num
    # tmp.loc[:, 'nxt'] = panos.lane_num.shift(1) == panos.lane_num
    # not_continuous = tmp[(tmp.prev | tmp.nxt) == False].Order.values.tolist()
    # not_continuous = tmp[(tmp.prev | tmp.nxt) == False].Order.values.tolist()

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
        df_tmp = _get_revert_df_edges(road_id, df_edges)
        matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_tmp, False) 
        matching = matching[atts].merge(df_tmp[['s', 'e']], left_on='osm_road_index', right_index=True)
        
    # 过滤异常值 -> 计算路段的统计属性
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

def matching_and_predicting_panos(RID_set, df_edges, vis=False):
    OSM_MATCHING_MEMO = {}
    for i in RID_set:
        OSM_MATCHING_MEMO[i] = OSM_MATCHING_MEMO.get(i, {})
        df = get_and_filter_panos_by_osm_rid(i, vis)
        OSM_MATCHING_MEMO[i]['df'] = df
        OSM_MATCHING_MEMO[i]['median'] = int(df.lane_num.median())

    return OSM_MATCHING_MEMO

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


#%%
# 初始化
# road_id = 529070115 # 打石一路
# road_id = 243387686 # 创科路
rid = road_id = 208128052 # 科技中二路

RID_set = [road_id, -road_id]
OSM_MATCHING_MEMO = matching_and_predicting_panos(RID_set, df_edges, vis=False)
# OSM_MATCHING_MEMO[-rid]['df']

#%% initial
SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'

tree = ET.parse('./osm_bbox.osm.bak.xml')
root = tree.getroot()
# 为osm里边的对象
osm_wayHash, osm_nodeHash = osm_parser(root)

"""粗调"""
for rid in RID_set:
    if rid < 0:
        continue
    lane = OSM_MATCHING_MEMO[rid]['median']
    item = osm_wayHash[rid]
    update_element_attrib(item['elem'], 'lanes', int(lane))


# internal functions
def _pre_process_fine_tune():
    """sumo releted process before fine tune
    """
    pre_process = f' rm -r ./{name}; mkdir {name}; cp {osm_file} ./{name}/{osm_file}; cd ./{name}'

    cmd = f"""
        {SUMO_HOME}/bin/netconvert  -t {SUMO_HOME}/data/typemap/osmNetconvert.typ.xml --geometry.remove --roundabouts.guess --ramps.guess -v --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --output.original-names --junctions.corner-detail 5 --output.street-names --tls.default-type actuated --osm-files {osm_file} --keep-edges.by-vclass passenger --offset.disable-normalization true -o {name}.net.xml
    """

    # create node, edge files
    cmd_tranfer0 = f"""{SUMO_HOME}/bin/netconvert --sumo-net-file {name}.net.xml --plain-output-prefix {name}; """
    for i in os.popen(' '.join( [pre_process, cmd, cmd_tranfer0] )).read().split('\n'): print(i)

def _post_process_fine_tune():
    indent(sumo_net['node_tree'].getroot())
    indent(sumo_net['edge_tree'].getroot())
    sumo_net['node_tree'].write('./osm/osm.nod.xml', encoding='utf-8')
    sumo_net['edge_tree'].write('./osm/osm.edg.xml', encoding='utf-8')

    """sumo releted process post fine tune
    """
    cmd_bck_net = f"""cd ./{name}; mv {name}.net.xml {name}_old.net.xml;
        {SUMO_HOME}/bin/netconvert --node-files {name}.nod.xml --edge-files {name}.edg.xml -t {name}.typ.xml  --precision 2 --precision.geo 6 --offset.disable-normalization true -o {name}.net.xml
    """ 
    post_precess = " cp ../start_with_net.sh ./; sh start_with_net.sh "
    for i in os.popen(' '.join( [cmd_bck_net, post_precess ] )).read().split('\n'): print(i)
    return

indent(root)
tree.write('osm_bbox.osm.xml', encoding='utf-8')
_pre_process_fine_tune()
# tranfer_to_sumo()


#%%
"""微调"""
sumo_net = parser_sumo_node_edge(name)
add_coords_to_osm_node_hash(osm_nodeHash, OSM_CRS)

#%%
def lane_seg_intervals(lane_num_dict):
    intervals = []
    for start, end, num in [(i, i+1, int(lane_num_dict[i])) for i in lane_num_dict ]:
        merge_intervals(intervals, start, end, num)

    return intervals

def get_road_changed_section(rid, vis=True):
    """获取变化的截面

    Args:
        rid ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    length_thres = 20

    def convert_interval_to_gdf(intervals):
        change_pids = gpd.GeoDataFrame(intervals, columns=['pano_idx_0', 'pano_idx_1', 'lane_num'])

        change_pids.loc[:, 'pano_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].geometry )
        change_pids.loc[:, 'pano_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].geometry )
        change_pids.loc[:, 'pano_id_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].PID )
        change_pids.loc[:, 'pano_id_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].PID )
        change_pids.loc[:, 'rid0'] = change_pids.pano_0.apply( lambda x: lines.loc[lines.distance(x).argmin()].start )
        change_pids.loc[:, 'rid1'] = change_pids.pano_1.apply( lambda x: lines.loc[lines.distance(x).argmin()].end )
        change_pids.loc[:, 'length'] = change_pids.apply( lambda x: x.pano_0.distance(x.pano_1)*110*1000, axis=1 )

        return change_pids


    # TODO add `lane_num`
    panos = OSM_MATCHING_MEMO[rid]['df']
    segments = panos.query(" lane_num != @ panos.lane_num.median() ")
    # 注意区间：左闭右开
    intervals = lane_seg_intervals(segments['lane_num'].to_dict())

    # gpd.GeoDataFrame({i: osm_nodeHash[i] for i in osm_wayHash[rid]['points']}).T.reset_index()
    # gpd.GeoDataFrame([ osm_nodeHash[i] for i in osm_wayHash[rid]['points']] )
    pids = osm_wayHash[rid]['points'] if rid > 0 else osm_wayHash[-rid]['points'][::-1]
    lines = gpd.GeoDataFrame([ {'index':i, 
                                'start': osm_nodeHash[pids[i]]['pid'],
                                'end': osm_nodeHash[pids[i+1]]['pid'],
                                'geometry': LineString( [osm_nodeHash[pids[i]]['xy'],
                                                        osm_nodeHash[pids[i+1]]['xy']] )} 
                                for i in range(len(pids)-1) ],
                            ).set_crs(epsg=4326)

    # second layer for filter
    change_pids = convert_interval_to_gdf((intervals))
    change_pids.query("length != 0", inplace=True)
    
    attrs = ['pano_idx_0', 'pano_idx_1', 'lane_num']

    keep      = change_pids.query(f"length >= {length_thres}")[attrs].values.tolist()
    candidate = change_pids.query(f"length < {length_thres}")[attrs].values.tolist()

    for i in candidate:
        keep = insert_intervals(keep, i)

    intervals = [i for i in keep if i not in candidate]

    change_pids = convert_interval_to_gdf((intervals))
    change_pids.loc[:, 'intervals'] = change_pids.apply( lambda x: [pids.index(x.rid0), pids.index(x.rid1)], axis=1 )
    
    return change_pids

def osm_road_segments_intervals(x, plst):
    def helpler(x):
        if 'cluster' in x:
            id = max( [plst.index(int(i)) for i in x.split("_")[1:] ] )
        else:
            id = plst.index(int(x))
        
        return id

    return [helpler(x['from']), helpler(x['to'])]

def lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst):
    lane_num_lst = [i[2] for i in new_intervals]
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
        
        sumo_net['edge_root'].append(elem)

def lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set):
    origin_start, origin_end = item.interval
    print(f"split intervals {item.id}, origin: [{origin_start}, {origin_end}], insert: [{new_start}, {new_end}]")

    # TODO split functions
    s = origin_start if pd.isnull(item.origFrom) else pids.index(int(item.origFrom))
    e = origin_end   if pd.isnull(item.origTo)   else pids.index(int(item.origTo))
    new_intervals = [[s        , new_start, int(item.numLanes)], 
                     [new_start, new_end  , lane_num_new], 
                     [new_end  , e        , int(item.numLanes)]
                    ]
    
    # check the distance of last interval 
    last_seg_dis = cal_dis_two_point( pids[new_end], pids[new_intervals[-1][1]])
    if last_seg_dis < dis_thres:
        _, end, _ = new_intervals.pop()
        new_intervals[-1][1] = end
    print(f"\tnew_intervals: {new_intervals}, the last segment dis is {last_seg_dis}")
    
    shape_lst = []
    for s, e, _ in new_intervals:
        _check_pano_in_SUMO_node(pids[s])
        _check_pano_in_SUMO_node(pids[e])
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
    id_lst = [ f"{item.rid}#{i}"  for i in order_lst ]
    
    print(f"id:\n\t{id_lst}")
    print(f"order:\n\t{order_lst}")
    print(f"new_intervals:\n\t{new_intervals}")
    print(f"\nshape_lst:\n\t {shape_lst}", )
    
    origin_edge = sumo_net['key_to_edge'][item.id]
    elem_lst = [origin_edge] + [copy.deepcopy(origin_edge) for _ in range(len(new_intervals)-1)]
    
    lane_change_process_for_node(elem_lst, pids, new_intervals, id_lst, shape_lst)

    for _, elem in enumerate(elem_lst):
        print("\n")
        print_elem(elem)
            

#%%
# osm_wayHash[rid]

def modify_road_shape(rid, dis_thres = 25):
    # rid = abs(rid)
    # rid= -rid
    change_pids = get_road_changed_section(rid)

    pids = osm_wayHash[rid]['points'] if rid > 0 else osm_wayHash[-rid]['points'][::-1]
    road = sumo_net['edge'].query(f"rid=={rid}").sort_values('order', ascending=True if rid >0 else False)
    road.loc[:, 'interval'] = road.apply(lambda x: osm_road_segments_intervals(x, pids), axis=1)
    # road = road[['id', 'from', 'to', 'numLanes', 'speed', 'shape', 'rid', 'order', 'interval']]
    order_set = set( road.order.values )

    queue = change_pids[['intervals', 'lane_num']].values.tolist()[::-1]
    while queue:
        [new_start, new_end], lane_num_new = queue.pop()

        for index, item in road.iterrows():
            origin_start, origin_end = item.interval
            
            if origin_start > new_end:
                # print(f'\tbreak: {item.id}')
                break
            elif origin_end < new_start:
                continue
            else:
                lane_change_process(item, new_start, new_end, dis_thres, pids, lane_num_new, order_set)
                # print('\t', item.id, new_start, new_end)
                break
            

for rid in RID_set:
    modify_road_shape(rid)

_post_process_fine_tune()



# %%
