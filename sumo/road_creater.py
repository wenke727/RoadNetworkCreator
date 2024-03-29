#%%
import os
import sys
import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
from shapely.geometry import LineString
from collections import deque
from tqdm import tqdm

from road_network import OSM_road_network
from utils.log_helper import LogHelper, logbook, log_type_for_sumo
from xml_helper import indent, update_element_attrib, print_elem
from utils.interval_helper import insert_intervals, merge_intervals

warnings.filterwarnings('ignore')

g_log_helper = LogHelper(log_dir="/home/pcl/traffic/RoadNetworkCreator_by_View/log", log_name='sumo.log')
SUMO_LOG = g_log_helper.make_logger(level=logbook.INFO)

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

OSM_CRS = None
OSM_MATCHING_MEMO = {}
matchingPanos = None

#%%

from sumo_helper import Sumo_Net, OSM_Net


# road creating related
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

        change_pids.loc[:, 'pano_0']    = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].geometry )
        change_pids.loc[:, 'pano_1']    = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].geometry )
        change_pids.loc[:, 'pano_id_0'] = change_pids.pano_idx_0.apply( lambda x: panos.loc[x].PID )
        change_pids.loc[:, 'pano_id_1'] = change_pids.pano_idx_1.apply( lambda x: panos.loc[x-1].PID )
        change_pids.loc[:, 'rid0']      = change_pids.pano_0.apply( lambda x: lines.loc[lines.distance(x).argmin()].start )
        change_pids.loc[:, 'rid1']      = change_pids.pano_1.apply( lambda x: lines.loc[lines.distance(x).argmin()].end )
        change_pids.loc[:, 'length']    = change_pids.apply( lambda x: x.pano_0.distance(x.pano_1)*110*1000, axis=1 )

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
    if change_pids.shape[0] > 1: 
        # when the length is 1, the func `std` return np.nan
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
    sumo_net.save(name)
    
    post_precess = f"cd ./{name}; ll; cp ../start_with_net.sh ./; sh start_with_net.sh "
    res = os.popen(post_precess).read()
    if verbose: print(res)
    
    if 'calling /usr/share/sumo/bin/duarouter' in res:
        flag = True
    
            
    return flag


# --------
osm_net = OSM_Net(file='./osm_bbox.osm.bak.xml', save_fn='./osm_bbox.osm.xml', logger=SUMO_LOG)
mathing_pano = MatchingPanos(df_edges)

road_types_lst = ['primary'] # 'trunk', 'secondary'
rids = osm_net.get_rids_by_road_levels(road_types_lst)


# osm_rid = 208128052 # 529249851
# mathing_pano.add(osm_rid)
mathing_pano.matching_lst(rids, vis=True, debug=True)


#%%


# %%
