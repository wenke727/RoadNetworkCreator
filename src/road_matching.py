#%%
import os
import pickle
from re import S
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from haversine import haversine
from shapely.geometry import Point
from tqdm import tqdm

from pano_img import get_staticimage, PANO_log, get_pano_ids_by_rid, traverse_panos_by_rid
from db.db_process import load_from_DB, extract_connectors_from_panos_respond, store_to_DB
from utils.geo_plot_helper import map_visualize
from utils.spatialAnalysis import linestring_length
from utils.utils import load_config
from utils.classes import Digraph, LongestPath
from utils.spatialAnalysis import *
from labels.label_helper import crop_img_for_lable
from road_network import OSM_road_network

# TODO 梳理函数名,感觉现在的很乱
# TODO 匹配完之后输出一张照片示意图，说明前行的方向

# def initialize():
#     global DB_pano_base, DB_panos, DB_connectors, DB_roads, config, pano_dir, pano_group_dir, DF_matching, osm_shenzhen, df_nodes, df_edges, connecters
DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
# connecters = extract_connectors_from_panos_respond( DB_pano_base, DB_roads )

config = load_config()
pano_dir = config['data']['pano_dir']
pano_group_dir = config['data']['pano_group_dir']
DF_matching = pd.read_csv( config['data']['df_matching'])

linestring_length(DB_roads, True)

# osm_shenzhen = pickle.load(open("../input/road_network_osm_nanshan.pkl", 'rb') )
osm_shenzhen = pickle.load(open("/home/pcl/traffic/data/input/road_network_osm_shenzhen.pkl", 'rb') )
df_nodes, df_edges = osm_shenzhen.nodes, osm_shenzhen.edges


# TODO initial; road network process
df_edges.reset_index(drop=True, inplace=True)
df_edges.loc[:,'rid'] = df_edges.loc[:,'rid'].astype(np.int)


#%%


def traverse_panos_in_district_level():
    """遍历所有links的街景图片
    """
    area_futian =  gpd.read_file( os.path.join(config['data']['input_dir'], 'Shenzhen_boundary_district_level_wgs.geojson') )
    area_futian = area_futian.loc[3].geometry

    links_ft = DB_roads[DB_roads.within(area_futian)]

    for RID in tqdm(links_ft.RID.values):
        traverse_panos_by_rid(RID, DB_panos, PANO_log)


def check_pid_duplication( folder = '/home/pcl/Data/minio_server/panos_data/Futian/益田路' ):
    # check whether there is any pid dulplicate in the folder 
    if not os.path.exists(folder): 
        os.mkdir(folder)
        return 
    
    lst = os.listdir(folder)
    df = pd.DataFrame(lst, columns=['fn'])
    df.loc[:, 'pid'] = df.fn.apply( lambda x: x.split("_")[-2] )

    df_count = pd.DataFrame(df.pid.value_counts())
    num = df_count.query('pid>1').shape[0]
    if num != 0:
        print( f"total num: {df.shape[0]}, dulpicate num: {df_count.query('pid>1').shape[0]}" )
    else:
        print( f"NO duplication: {folder}" )
    return


def _matching_panos_path_to_network( road, DB_roads=DB_roads, vis=True, vis_step=False, save_fig=True, buffer_thres = 0.00005, angel_thres = 30):
    """
    find the matching path of panos for a special road based on the frechet distance
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
    road_candidates.loc[:, 'frechet_dis']    = res_dis
    road_candidates.loc[:, 'angel']          = res_ang
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


def get_panos_of_road_by_id(road_id, df_edges, vis=False, save=False):
    # ! # FIXME
    """通过frnchet距离匹配某条道路的百度街景pano轨迹，并输出

    Args:
        road_id ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    dis_thes = 50; angle_thre = 30

    road_osm = df_edges.query( f"rid == {road_id}" )
    
    # 可视化
    # fig, ax = map_visualize(road_osm, scale=0.1)
    # road_osm.plot(column='index', cmap='jet', ax=ax, legend=True)
    
    linestring_length(road_osm, True)

    result = []
    for i in range(road_osm.shape[0]):
        road_candidates, rid = _matching_panos_path_to_network( road_osm.iloc[i], vis=False, vis_step=False )
        if road_candidates is not None:
            result.append(road_candidates)
    
    if len(result) <=0:
        return None
    
    matching = pd.concat(result).reset_index(drop=True)
    # filter result
    matching.query( f" frechet_dis < {dis_thes} and angel < {angle_thre}", inplace=True )
    matching.drop_duplicates('RID', keep ='first', ignore_index=True, inplace=True)
    
    # 看下顺序是否满足要求
    if vis:
        fig, ax = map_visualize(road_osm, scale=0.01, lyrs='s', color='black', figsize=(10,10), label='road line')
        matching.plot(ax=ax, color='red', linestyle="-.", label='matching panos')
        plt.legend()

    if save:
        matching.to_file( f'../output/tmp_road_match_{road_id}_{id}.geojson', driver="GeoJSON" )
    
    return matching


# def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
#     """obtain the panos in road[rid] 

#     Args:
#         rid (str): the id of road segements

#     Returns:
#         [type]: [description]
#     """
    
#     df_pids = get_pano_ids_by_rid(rid, DB_panos)
    
#     pano_lst = df_pids[['Order','PID', 'DIR']].values
#     length = len(pano_lst)
#     res, pre_heading = [], 0
    
#     for id, (order, pid, heading) in enumerate(pano_lst):
#         if heading == 0 and id != 0:   # direction, inertial navigation
#             heading = pre_heading
        
#         if not all:
#             if length > 3 and order % 3 != 1:
#                 # print(order)
#                 continue

#         fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
#         res.append(get_staticimage(pid=pid, heading=heading, path=fn, log_helper=log))
#         pre_heading = heading
        
#     return res, df_pids



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
    """获取匹配的id

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

    res = pd.DataFrame({'road_id': [road_id] *len(res) ,'path':res})
    res.reset_index(inplace=True)

    
    df_matching = df_matching.append(res, ignore_index=True)
    df_matching.to_csv( df_matching_path )
    
    fn = group_panos_by_road( road_id, res )
    return 


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


def get_panos_of_road_and_indentify_lane_type_by_id( road_id, df_edges, vis=False, save=False, len_thres=50):
    
    print("\tget_panos_of_road_and_indentify_lane_type_by_id: ",road_id)
    matching = get_panos_of_road_by_id(road_id, df_edges, vis, save)
    if matching is None: return None

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
        path = [ (sol.path[i+1], sol.path[i]) for i in range( len(sol.path)-1)] \
             + [ (sol.path[1], sol.path[i+1]) for i in range( len(sol.path)-1)]

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


def crawl_pano_imgs_by_roadid_batch(road_ids, df_edges, road_name, visited=set([])):
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
    matching.sort_values( by = ['link', 'group_num','index'], ascending =[True, True, True], inplace=True )
    matching.drop_duplicates('RID', keep ='first', ignore_index=True, inplace=True)

    # obtain panos imgs
    panos_img_paths = []; road_type_lst = []
    for rid, road_type in tqdm( matching[['RID', 'link']].values, desc="traverse panos by rid" ):
        fns, _ = traverse_panos_by_rid(rid=rid, DB_panos=DB_panos, log=PANO_log)
        panos_img_paths += fns
        road_type_lst += [road_type] * len(fns)

    # group imgs by road segemts and copy to another folder
    panos_img_paths = pd.DataFrame({'road_id': ['_'.join(map(str, road_ids))] *len(panos_img_paths), 'road_type':road_type_lst,'path':panos_img_paths})
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
    """[summary]

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


# %%
if __name__ == '__main__':
    
    # traverse_panos_in_district_level()
    # traversed_lst = "福华三路"
    """ 福田区 """
    # lst = [
    #     '香蜜湖路', '香梅路', '皇岗路', '福田路', '民田路', '福田中心四路', '福田中心五路',  '红树林路',
    #     '福强路', '福民路', '福华一路', '福中路', '福中一路', '深南中路', '红荔路', '红荔西路', '莲花路', '笋岗西路', '侨香路'
    # ]
    
    lst = ['民田路', '益田路', '福中一路'
            '福华一路',
            '福华路',
            '福民路',
            '福田路',
            '笋岗西路',
            '红树林路',
            '红荔西路',
            '莲花路',
            '金田路']
    error_lst = []
    
    roads = "文锦北路、文锦中路、文锦南路、沙湾路、太白路、东门北路、东门南路、嘉宾路、江背路、和平路、建设路、春风路、人民南路、友谊路、宝安北路、宝安南路、罗沙路、莲塘路、国威路、聚宝路、新秀路、布心路、东湖路、翠竹路、贝丽北路、贝丽南路、水贝一路、水贝二路、田贝一路、田贝二路、田贝三路、田贝四路、湖贝路、乐园路、中兴路、南湖路、晒布路、新园路、松园路、红桂路、红宝路、解放路、金塘路、书城路、万象路、向西路、迎春路、永新路、南庆路、东升路、立新路、人民北路、红岗路、凤凰路、凤翔路、清平路、港莲路、华丽路、金湖路、罗芳路、桂园路、新安路、兴湖路、蛟湖路、洪湖西路、洪湖一路、宝岗路、人民公园路、蔡屋围一路、莲罗路、三号支路、碧波一路、金稻田路、金碧路、聚财路、太宁路、童乐路、莲十路、望桐路、大望大道、新田大道、深南辅道、交通楼三楼、沿河路、延芳路、蜜园路、梨园路、展艺路、柑园路、祝福路、迎宾东路、交通楼一楼、黄贝路、沿河南路、沿河北路、罗湖口岸交通楼、环仓路、北斗路、船步路、翠云路、清水河一路、清水河二路、清水河三路、清水河四路、清水河五路、太安路、东晓路、东昌路、翠茵路、河西环路、洪湖二路、洪湖五路、桃园路、梅园路、北站路、松园南路、松园北路、桂园北路、果园路、果园东路、煤场路、红桂二路、红桂横路、桃丽路、笋田一路、西货场路、河边路、锦湖路、银湖路、迎宾西路、鹏兴路、翠园路、仙湖路、畔山路、怡景路、东门老街、望桐新路、水田一路、水田二路、泥岗路、笋中路、东湖二路、嘉北路、嘉南路、翠山路、鸿业路、东乐路、金洲路"    
    
    lst = ['吉华路']
    # lst = roads.split("、")
    for fn in lst:
        try:
            folder = f'/home/pcl/Data/minio_server/panos_data/Longgang/{fn}'
            if os.path.exists(folder):
                f_lst = os.listdir(folder)
                for f in f_lst:
                    if 'jpg' not in f:
                        continue
                    tmp = f" rm { os.path.join(folder, f) }"
                    os.popen( tmp )

            print(fn)
            get_pano_imgs_of_road_by_name(fn)
            check_pid_duplication(folder)
        except:
            error_lst.append(fn)
    print("crawl failed", error_lst)
    
    pass


if False:
    
    # %%
    road_name = '打石一路'

    roads = df_edges.query( f"name == '{road_name}' " )

    # TODO 有一些分叉需要处理掉
    matching_0 = get_panos_of_road_by_id(362735582, df_edges, True)
    matching_1 = get_panos_of_road_by_id(529070115, df_edges, True)

    matching1 = get_panos_of_road_and_indentify_lane_type_by_id(362735582, df_edges, True)
    matching  = get_panos_of_road_by_id(362735582, df_edges, True)

    road_id = 362735582
    
    # %%

# TODO 整理代码
# road_id = 362735582

# road_osm.iloc[4]

# rid = '42fe8d-9555-1595-7843-afe68c'

# panos = get_pano_ids_by_rid(rid, DB_panos)

# lane_shape_predict_memo = '/home/pcl/Data/minio_server/input/lane_shape_predict_memo.csv'
# df_pred_memo = pd.read_csv(lane_shape_predict_memo)
# df_pred_memo.loc[:, 'pred'] = df_pred_memo.pred.apply(lambda x: eval(x))
# # df_pred_memo.loc[:,'PID'] = df_pred_memo.name.apply(lambda x: x.split('_')[-2])
# # df_pred_memo.loc[:,'DIR'] = df_pred_memo.name.apply(lambda x: x.split('_')[-1].split('.')[0]).astype(np.int)
# # df_pred_memo[['PID', 'DIR', 'lane_num', 'name',  'pred']].to_csv( lane_shape_predict_memo, index=False )


# panos = panos.merge( df_pred_memo, on=["PID", 'DIR'] )

# from model.lstr import draw_pred_lanes_on_img

# tmp = panos.iloc[0].to_dict()
# tmp['file'] = tmp['name']
# tmp['pred'] = eval(tmp['pred'])

# draw_pred_lanes_on_img(tmp, 'tmp.jpg')


# for i in range( panos.shape[0] ):
#     tmp = panos.iloc[i].to_dict()
#     tmp['file'] = tmp['name']
#     draw_pred_lanes_on_img(tmp, f'./tmp/{i:02d}.jpg')


# # 保存所有的记录
# # DB_panos = DB_panos.merge( df_pred_memo[["PID", 'DIR','lane_num']], on=["PID", 'DIR'], how='left' ).fillna(-1)
# # DB_panos.lane_num.value_counts()
# # store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)


# # 更新最后一个节点
# df_last_point = DB_panos[['RID','Order']].groupby('RID').max().reset_index().query('Order > 0').rename(columns={'Order':'max_ord'})
# df_last_point.loc[:, 'sec_ord'] = df_last_point.max_ord - 1
# df_last_point = df_last_point.merge( DB_panos[['RID','Order','DIR']], left_on=['RID', 'sec_ord'], right_on=['RID',"Order"] )
# df_last_point.drop(columns='Order', inplace=True)
# df_last_point.rename(columns={'DIR': 'dir'}, inplace=True)


# DB_panos.loc[:, 'DIR_bak'] = DB_panos.DIR
# cols = DB_panos.columns

# DB_panos = DB_panos.merge(df_last_point, left_on=['RID',"Order"], right_on=['RID', 'max_ord'], how='left')
# DB_panos.loc[:, 'DIR'] = (DB_panos.DIR + DB_panos.dir.fillna(0)).astype(np.int)
# DB_panos = DB_panos[cols]

# DB_panos.query("RID == '69c977-b392-e17d-abc7-1b754f' ")

# store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
