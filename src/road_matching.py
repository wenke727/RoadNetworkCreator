#%%
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from haversine import haversine
from shapely.geometry import Point
from tqdm import tqdm

from pano_img import PANO_log, traverse_panos_by_rid
from db.db_process import load_from_DB, extract_connectors_from_panos_respond, store_to_DB
from db.features_API import get_features
from utils.geo_plot_helper import map_visualize
from utils.spatialAnalysis import linestring_length
from utils.df_helper import query_df
from utils.utils import load_config
from utils.classes import Digraph, LongestPath
from utils.spatialAnalysis import *
from labels.label_helper import crop_img_for_lable
from road_network import OSM_road_network

import warnings
warnings.filterwarnings('ignore')

# TODO 梳理函数名,感觉现在的很乱
_, DB_panos, DB_connectors, DB_roads = load_from_DB(False)

config = load_config()
pano_dir = config['data']['pano_dir']
pano_group_dir = config['data']['pano_group_dir']

df_edges = get_features('edge')

#%%
# 辅助函数
def _matching_panos_path_to_network( road, DB_roads=DB_roads, vis=True, vis_step=False, save_fig=True, buffer_thres=0.00005, angel_thres=30):
    """Find the matching path of panos for a special road based on the frechet distance

    Args:
        road ([type]): [description]
        DB_roads ([type], optional): [description]. Defaults to DB_roads.
        vis (bool, optional): [description]. Defaults to True.
        vis_step (bool, optional): [description]. Defaults to False.
        save_fig (bool, optional): [description]. Defaults to True.
        buffer_thres (float, optional): [description]. Defaults to 0.00005.
        angel_thres (int, optional): [description]. Defaults to 30.

    Returns:
        [road_candidates: dataframe]: [description]
        [rid]: matching rid of osm road
    """
    # road = road_osm.iloc[0]; vis=True; vis_step=True; buffer_thres = 0.00005; angel_thres = 30; save_fig=True
    
    road_candidates = DB_roads[ DB_roads.intersects( road.geometry.buffer(buffer_thres) )].query( 'length > 0' )
    if road_candidates.shape[0] <=0:
        return None, None
    
    res_dis, res_ang = [], []
    for index, item in road_candidates.iterrows():
        if item.length == 0:
            res_dis.append(np.inf)
            res_ang.append(90)
            continue
        
        # 若是两条线几乎垂直，可以考虑忽略了
        angel = angle_bet_two_linestring_ignore_inte_point(item, road)
        res_ang.append(angel)

        if 90-angel_thres< angel < 90 + angel_thres:
            res_dis.append(float('inf'))    
        else:
            l0, l1  = cut_and_align( item.geometry, road.geometry )
            l0, l1  = line_interplation(l0), line_interplation(l1)
            dis, dp = frechet_distance_bet_polyline( l0, l1 )
            res_dis.append( dis *110*1000 )

            if not vis_step:
                continue

            fig, ax = map_visualize( gpd.GeoSeries( [ road.geometry ] ), color='black', label='OSM road' )
            gpd.GeoSeries( [ item.geometry ] ).plot(color='gray', label='Pano path', ax=ax )
            for line in [l0, l1]: gpd.GeoSeries([ Point(x) for x in line.coords[:]]).plot(ax=ax)
            plt.title( f"frechet dis: {dis*110*1000:.2f}" )
            plt.legend()

    # 汇总统计结果 
    road_candidates.loc[:, 'angel']          = res_ang
    road_candidates.loc[:, 'frechet_dis']    = res_dis
    road_candidates.loc[:, 'osm_road_id']    = road.rid
    road_candidates.loc[:, 'osm_road_index'] = road.name
    road_candidates.loc[:, 'related_pos']    = road_candidates.geometry.apply( lambda x: get_related_position(x, road.geometry) )
    road_candidates.sort_values(by='related_pos', inplace=True)
    
    for att in ["osm_road_index", "osm_road_id"]:
        road_candidates.loc[:, att] = road_candidates.loc[:, att].astype(np.int)
    
    rid = road_candidates.loc[road_candidates.frechet_dis.idxmin()].RID

    if vis:
        fig, ax = map_visualize( road_candidates, color='black', label='Pano paths', linestyle=':' )
        for index, item in road_candidates.iterrows():
            ax.text(*item.geometry.centroid.coords[0], 
                    f"{item.frechet_dis:.0f}, {item.angel:.0f},\n {item.related_pos:.2f}",
                    horizontalalignment='center', verticalalignment='center' 
                    )

        gpd.GeoSeries( [ road.geometry ] ).plot( color='red', label="OSM road", ax=ax )
        road_candidates.query( f"RID=='{rid}'" ).plot( color='blue', linestyle='--' , label = "match pano", ax=ax )
        plt.legend()
        
        if save_fig: plt.savefig( f'../log/match_process/{road.name}.jpg', pad_inches=0.1, bbox_inches='tight' )

    return road_candidates, rid


def _get_links_by_pids(pids:list, connecters:gpd.GeoDataFrame, cal_length=True):
    links = connecters.query( f"prev_pano_id in {pids}" )
    tmp = links.merge( DB_panos[['PID', 'geometry']], left_on='prev_pano_id', right_on='PID' )

    if tmp.shape[0] == 0:
        return None
    # print(tmp.apply( lambda x: haversine(x.geometry_x.coords[:][0], x.geometry_y.coords[:][0]), axis=1))
    
    if cal_length:
        links.loc[:,'length'] = tmp.apply( lambda x: haversine(x.geometry_x.coords[:][0], x.geometry_y.coords[:][0]), axis=1)*1000
    del tmp
    
    links = links[['prev_pano_id', 'PID', 'length']].rename( columns={"prev_pano_id": 'PID_start', 'PID':'PID_end'} )
    links.loc[:, 'link'] = True
     
    return links


def _get_road_origin_points(df_roads):
    """get the origin points, with 0 indegree and more than 1 outdegree, of the roads

    Args:
        df_roads (pd.Datafrem): With attributes `start` and `end`

    Returns:
        origins [list]: The coordinations of origins.
    """
    node_dic = {}
    count = 0

    for i in np.concatenate( [df_roads.start, df_roads.end] ):
        if i in node_dic:
            continue
        node_dic[i] = count
        count += 1

    node = pd.DataFrame([node_dic], index=['id']).T
    edges = df_roads.merge( node, left_on='start', right_index=True ).merge( node, left_on='end', right_index=True, suffixes=['_0', '_1'] )
    node = node.reset_index().rename(columns={"index": 'coord'}).set_index('id')
    
    network = Digraph( edges = edges[['id_0', 'id_1']].values )
    origins = network.get_origin_point()

    return [ node.loc[i, 'coord'] for i in  origins]


# 道路匹配
def get_panos_of_road_by_id(road_id, df_edges, vis=False, save=False, dis_thes = 50, angle_thre = 30):
    """通过frenchet距离匹配某条道路的百度街景pano轨迹，并返回匹配的pano

    Args:
        road_id ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [pd.DataFrame]: 数据表
    """

    road_osm = df_edges.query( f"rid == {road_id}" )
    linestring_length(road_osm, True)
    
    # 可视化
    # fig, ax = map_visualize(road_osm, scale=0.1)
    # road_osm.plot(column='index', cmap='jet', ax=ax, legend=True)

    result = []
    for i in range(road_osm.shape[0]):
        road_candidates, rid = _matching_panos_path_to_network( road_osm.iloc[i], vis=False, vis_step=False )
        if road_candidates is None:
            continue
        result.append(road_candidates)
    
    if len(result) <=0:
        return None
    
    # filter result
    matching = pd.concat(result).reset_index(drop=True)
    matching.query( f" frechet_dis < {dis_thes} and angel < {angle_thre}", inplace=True )
    matching.drop_duplicates('RID', keep ='first', ignore_index=True, inplace=True)
    
    if vis:
        fig, ax = map_visualize(road_osm, scale=0.01, lyrs='s', color='black', figsize=(10,10), label='road line')
        matching.plot(ax=ax, color='red', linestyle="-.", label='matching panos')
        plt.legend()

    if save:
        matching.to_file( f'../output/tmp_road_match_{road_id}_{id}.geojson', driver="GeoJSON" )
    
    return matching


def get_panos_of_road_and_indentify_lane_type_by_id( road_id, df_edges, vis=False, save=False, len_thres=50, verbose=False):
    """在get_panos_of_road_by_id的基础上，针对匹配的panos进行分析；
    1. group_panos_by_road 获取基础的pano
    1. 获取links -> 构建grap; 
    1. LongestPath 获取最长的路径，但`这个不太科学`, # FIXME

    Args:
        road_id ([type]): [description]
        df_edges ([type]): [description]
        vis (bool, optional): [description]. Defaults to False.
        save (bool, optional): [description]. Defaults to False.
        len_thres (int, optional): [description]. Defaults to 50.

    Returns:
        mathcing panos: The matching panos in geodataframe
    """
    
    if verbose: print("\tget_panos_of_road_and_indentify_lane_type_by_id: ",road_id)
    matching = get_panos_of_road_by_id(road_id, df_edges, vis, save)
    if matching is None: return None

    connecters = DB_connectors.copy()
    att_lst = ['PID_start', 'PID_end', 'length']
    a = _get_links_by_pids( matching.PID_end.values.tolist(), connecters )
    b = DB_roads.query( f"PID_start in {a.PID_end.values.tolist()} and PID_start == PID_end" )[att_lst] if a is not None else None
    c = _get_links_by_pids( b.PID_end.values.tolist(), connecters ) if b is not None else None
    d = matching[att_lst]

    edges = pd.concat( [a, c, d] ).fillna(False)
    edges.reset_index(drop=True, inplace=True)
    edges.rename( columns={'PID_start':'start', 'PID_end':'end'}, inplace=True )

    # obtain and sort origins 
    origins = _get_road_origin_points( edges )
    origins = matching.query( f"PID_start in {origins}").PID_start.values

    main_road = []
    for origin in origins:
        sol = LongestPath( edges, origin )
        # ? 猜测 双向
        path = [ (sol.path[i+1], sol.path[i]) for i in range( len(sol.path)-1)] \
             + [ (sol.path[i], sol.path[i+1]) for i in range( len(sol.path)-1)]

        # TODO threding惯性
        if sol.length > len_thres:
            path = pd.DataFrame(path, columns=['start', 'end'])
            main_road.append(path)

            visited = path.apply( lambda x: x.start+";"+x.end, axis=1 ).values.tolist()
            con = edges.apply( lambda x: x.start+";"+x.end in visited, axis=1 )
            edges = edges[~con]
        
            if not vis: continue
            map_visualize( DB_roads.merge( path, left_on=['PID_start','PID_end'], right_on=['start','end'] ) )

    if len(main_road) == 0: return None
    
    main_road = pd.concat(main_road)
    visited = main_road.apply( lambda x: x.start+";"+x.end, axis=1 ).values.tolist()
    con = matching.apply( lambda x: x.PID_start+";"+x.PID_end in visited, axis=1 )
    
    matching.loc[~con, 'link'], matching.loc[con, 'link'] = 1, 0
    matching.loc[:, 'link'] = matching.loc[:, 'link'].astype(np.int)
    
    return matching.reset_index() 


# 街景数据下载
def group_panos_by_road(road_id, df_matching, df_edges=df_edges):
    """将已下载的panos根据匹配的道路重组，并复制到特定的文件夹中

    Args:
        road_id ([type]): [description]
        df_matching ([type]): [description]
        df_edges ([type], optional): [description]. Defaults to df_edges.
    """
    import shutil
    fn_lst = df_matching.query(f'road_id == {road_id}').path.values
    road_name = df_edges.query( f"rid=={road_id}" ).name.unique()[0]
    folder = os.path.join( pano_group_dir, f"{road_name}_{road_id}" )

    if not os.path.exists(folder): 
        os.mkdir(folder)

    for fn in fn_lst:
        shutil.copy( fn, folder )

    return folder


def crawl_pano_imgs_by_roadid(road_id, df_edges, df_matching_path=config['data']['df_matching']):
    """通过pid爬取相应的pano

    Args:
        road_id ([type]): [description]
    """
    df_matching = pd.read_csv(df_matching_path)
    if road_id in df_matching.road_id.unique():
        return
    
    matching = get_panos_of_road_by_id( road_id,df_edges )
    if matching is None:
        return 

    res = []
    for rid in tqdm( matching.RID.values ):
        r, _ = traverse_panos_by_rid(rid=rid, DB_pano=DB_panos,log=PANO_log)
        res += r

    res = pd.DataFrame({'road_id': [road_id] *len(res), 'path':res})
    res.reset_index(inplace=True)
    
    df_matching = df_matching.append(res, ignore_index=True)
    df_matching.to_csv( df_matching_path )
    
    fn = group_panos_by_road( road_id, res )

    return 


def crawl_pano_imgs_by_roadid_batch(road_ids, df_edges, road_name, visited=set([])):
    """通过pid爬取相应的pano batch 版本, crawl_pano_imgs_by_roadid

    Args:
        road_ids ([type]): [description]
        df_edges ([type]): [description]
        road_name ([type]): [description]
        visited ([type], optional): [description]. Defaults to set([]).

    Returns:
        [type]: [description]
    """
    # road_ids = [183920405,]; road_name="TEST"
    matching_lst = []
    for i, id in enumerate(road_ids):
        # add att `link` to split matching into two part: 1) main road; 2) links
        matching = get_panos_of_road_and_indentify_lane_type_by_id( id, df_edges )
        # matching = get_panos_of_road_by_id( id, df_edges )
        if matching is None: 
            continue
        matching.loc[:, 'group_num'] = i
        matching_lst.append(matching)

    if len(matching_lst) == 0: return visited
    
    matching = pd.concat(matching_lst)
    matching.sort_values( by=['link', 'group_num','index'], ascending =[True, True, True], inplace=True )
    matching.drop_duplicates('RID', keep ='first', ignore_index=True, inplace=True)

    # obtain panos imgs
    panos_img_paths = []; road_type_lst = []
    for rid, road_type in tqdm( matching[['RID', 'link']].values, desc="traverse panos by rid" ):
        fns, _ = traverse_panos_by_rid(rid=rid, DB_panos=DB_panos, log=PANO_log)
        panos_img_paths += fns
        road_type_lst += [road_type] * len(fns)

    # group imgs by road segemts and copy to another folder
    panos_img_paths = pd.DataFrame({'road_id': ['_'.join(map(str, road_ids))] *len(panos_img_paths), 'road_type':road_type_lst, 'path':panos_img_paths})
    panos_img_paths.reset_index(inplace=True)
    panos_img_paths.loc[:,'pid'] = panos_img_paths.path.apply( lambda x: x.split("_")[-2] )
    panos_img_paths.drop_duplicates('pid', keep ='first', ignore_index=True, inplace=True)

    # if panos_img_paths.query( f"pid in {list(visited)}").shape[0] > 0:
    #     print( "panos_img_paths.query, pid in visited", panos_img_paths.query( f"pid in {list(visited)}").shape[0] )
    panos_img_paths.query( f"pid not in {list(visited)}", inplace=True )

    fn_lst = panos_img_paths[['path','road_type']].values
    folder = os.path.join( pano_group_dir, f"{road_name}" )
    if not os.path.exists(folder):os.mkdir(folder)
    
    for index, item in enumerate(fn_lst):
        fn, road_type = item
        try:
            crop_img_for_lable(fn,  os.path.join(folder, f"{'' if road_type == 0 else 'links_' }{road_ids[0]}_{index:03d}_{fn.split('/')[-1]}"), False )
        except:
            print(f"error: {fn}")

    visited = visited.union( panos_img_paths.pid.values.tolist() )
    print(f"****** visited: {len(visited)}")
    return visited


def get_pano_imgs_of_road_by_name(road_name, df_edges=df_edges):
    """通过 道路名 查询，然后抓取panos

    Args:
        road_name ([type]): [description]
        df_edges ([gpd.GeoDataFrame], optional): [description]. Defaults to df_edges.
    """

    roads = df_edges.query( f"name == '{road_name}' " )
    map_visualize(roads)
    network = Digraph(edges=roads[['s', 'e']].values)

    result = network.combine_edges(roads)
    visited = set([])
    
    for _, road_ids in result:
        print(road_ids)
        visited = crawl_pano_imgs_by_roadid_batch(road_ids, df_edges, road_name, visited)


    return 


def start_crawle_panos(lst = ['民田路', '益田路', '金田路']):
    error_lst = []
    for fn in lst:
        try:
            folder = f'/home/pcl/Data/minio_server/tmp/{fn}'
            if os.path.exists(folder):
                f_lst = os.listdir(folder)
                for f in f_lst:
                    if 'jpg' not in f:
                        continue
                    tmp = f" rm { os.path.join(folder, f) }"
                    os.popen( tmp )

            print(fn)
            get_pano_imgs_of_road_by_name(fn)
        except:
            error_lst.append(fn)
    print("crawl failed", error_lst)
    
    pass    

# %%
if __name__ == '__main__':
    """ 通过路名抓取数据 """
    start_crawle_panos(['南光路'])

    # 通过道路名称来匹配, 整个道路的数量情况

    """ 道路匹配 """    
    road_name = '打石一路'
    rois = query_df(df_edges, 'name', road_name)
    # 基础方案
    matching_0 = get_panos_of_road_by_id(362735582, rois, vis=True)
    # 处理掉一些分叉，但效果较一般
    matching_1 = get_panos_of_road_and_indentify_lane_type_by_id(362735582, rois, True)

    lane_num = matching_0.lane_num.mode().values[0]

    # 科技中二路
    road_id = 208128052
    matching1 = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, True)
        

    pass
    
    # %%


