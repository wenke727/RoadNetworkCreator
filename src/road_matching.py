#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely import geometry
from shapely.geometry import Point
from tqdm import tqdm

from pano_img import get_staticimage, traverse_panos_by_rid, PANO_log, get_pano_ids_by_rid
from db.db_process import load_from_DB
from utils.geo_plot_helper import map_visualize
from utils.spatialAnalysis import linestring_length
from utils.utils import load_config
from utils.classes import Digraph
from utils.spatialAnalysis import *
from labels.label_helper import crop_img_for_lable
import pickle
from road_network import OSM_road_network

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)

config = load_config()
pano_dir = config['data']['pano_dir']
pano_group_dir = config['data']['pano_group_dir']
DF_matching = pd.read_csv( config['data']['df_matching'])

linestring_length(DB_roads, True)

# osm_shenzhen = pickle.load(open("../input/road_network_osm_nanshan.pkl", 'rb') )
osm_shenzhen = pickle.load(open("/home/pcl/traffic/data/input/road_network_osm_shenzhen.pkl", 'rb') )
df_nodes, df_edges = osm_shenzhen.nodes, osm_shenzhen.edges

# TODO initial
df_edges.reset_index(drop=True, inplace=True)
df_edges.loc[:,'rid'] = df_edges.loc[:,'rid'].astype(np.int)


def matching_panos_path_to_network( road, DB_roads=DB_roads, vis=True, vis_step=False, save_fig=True, buffer_thres = 0.00005, angel_thres = 30):
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


def get_panos_of_road_by_id(road_id, df_edges,vis=False, save=False):
    """通过frnchet距离匹配某条道路的百度街景pano轨迹，并输出

    Args:
        road_id ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    dis_thes = 50; angle_thre = 30
    result = []

    road_osm = df_edges.query( f"rid == {road_id}" )
    linestring_length(road_osm, True)

    for i in range(road_osm.shape[0]):
        road_candidates, rid = matching_panos_path_to_network( road_osm.iloc[i], vis=False, vis_step=False )
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
        fig, ax = map_visualize(road_osm, scale=0.01, lyrs='s', color='black', figsize=(20,20))
        matching.plot(ax=ax, color='red', linestyle="-.")

    if save:
        matching.to_file( f'../output/tmp_road_match_{road_id}_{id}.geojson', driver="GeoJSON" )
    
    return matching


def traverse_panos_by_rid(rid, log=None):
    """obtain the panos in road[rid] 

    Args:
        rid (str): the id of road segements

    Returns:
        [type]: [description]
    """
    
    df_pids = get_pano_ids_by_rid(rid)
    pano_lst = df_pids[['Order','PID', 'DIR']].values
    res, pre_heading = [], 0
    # FIXME the strategy to crawl the panos data
    
    for id, (order, pid, heading) in enumerate(pano_lst):
        # direction, inertial navigation
        if heading == 0 and id != 0:  
            heading = pre_heading
        
        fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        res.append(get_staticimage(pid=pid, heading=heading, path=fn, log_helper=log))
        pre_heading = heading
        
    return res


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
        # TODO 截取
        shutil.copy( fn, folder )

    return folder


def get_pano_imgs_of_road_by_id(road_id, df_edges, df_matching_path=config['data']['df_matching']):
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
        res += traverse_panos_by_rid(rid=rid, log=PANO_log)

    res = pd.DataFrame({'road_id': [road_id] *len(res) ,'path':res})
    res.reset_index(inplace=True)

    
    df_matching = df_matching.append(res, ignore_index=True)
    df_matching.to_csv( df_matching_path )
    
    fn = group_panos_by_road( road_id, res )
    return 


def get_pano_imgs_of_road_by_id_batch(road_ids, df_edges, road_name):
    # matching paths
    matching_lst = []
    for id in road_ids:
        matching = get_panos_of_road_by_id( id, df_edges )
        if matching is None:
            continue
        matching_lst.append(matching)

    if len(matching_lst) == 0:
        return False
    
    matching = pd.concat(matching_lst)
    matching.drop_duplicates('RID', keep ='first', ignore_index=True, inplace=True)

    # obtain panos imgs
    panos_img_paths = []
    for rid in tqdm( matching.RID.values ):
        panos_img_paths += traverse_panos_by_rid(rid=rid, log=PANO_log)

    panos_img_paths = pd.DataFrame({'road_id': ['_'.join(map(str, road_ids))] *len(panos_img_paths) ,'path':panos_img_paths})
    panos_img_paths.reset_index(inplace=True)
    panos_img_paths.drop_duplicates('path', keep ='first', ignore_index=True, inplace=True)

    # group imgs by road segemts and copy to another folder
    fn_lst = panos_img_paths.path.values
    # folder = os.path.join( pano_group_dir, f"{road_name}_{'_'.join(map(str, road_ids))}" )
    folder = os.path.join( pano_group_dir, f"{road_name}" )
    if not os.path.exists(folder):  os.mkdir(folder)
    for index, fn in enumerate(fn_lst):
        # shutil.copy( fn, folder)
        try:
            crop_img_for_lable(fn,  os.path.join(folder, f"{road_ids[0]}_{index:03d}_{fn.split('/')[-1]}"), False )
        except:
            print(f"error: {fn}")

    return True


def traverse_panos_in_district_level():
    """遍历福田区域所有links的街景图片
    """
    area_futian =  gpd.read_file( os.path.join(config['data']['input_dir'], 'Shenzhen_boundary_district_level_wgs.geojson') )
    area_futian = area_futian.loc[3].geometry

    links_ft = DB_roads[DB_roads.within(area_futian)]

    for RID in tqdm(links_ft.RID.values):
        traverse_panos_by_rid(RID, PANO_log)


def main(road_name):
    roads = df_edges.query( f"name == '{road_name}' " )
    network = Digraph(edges=roads[['s', 'e']].values)

    result = network.combine_edges(roads)

    for _, road_ids in result:
        print(road_ids)
        get_pano_imgs_of_road_by_id_batch(road_ids, df_edges, road_name)

    map_visualize(roads)

    return 

#%%

if __name__ == '__main__':
    
    # traverse_panos_in_district_level()
    # traversed_lst = "福华三路"
    main("福中路")
    main("福强路")
    main("福强路")
    main("金田路")
    
    
    """ 旧方法：通过路段的ID来匹配 """
    # road_name = '打石一路'
    # map_visualize( df_edges.query( f"name == '{road_name}' " ), lyrs='s', scale=0.01 )
    # road_ids = df_edges.query( f"name == '{road_name}' " ).rid.unique()
    # # get_pano_imgs_of_road_by_id(road_id)
    # get_pano_imgs_of_road_by_id(road_ids[0])
    # main("北环大道")
    
    # road_name = '香蜜湖路'
    
    # lst_nanshan = [
    #     #    "深南大道",
    #        "月亮湾大道", 
    #        "南山大道", 
    #        "科苑大道", 
    #        "沙河西路", 
    #        "沙河东路", 
    #        "滨海大道",
    #        "留仙大道", 
    #        "中山园路", 
    #        "工业大道", 
    #        "南油大道", 
    #        "麒麟路",
    #        "同发路",
    #        "侨城东路", 
    #        "茶光路", 
    #        "龙珠大道", 
    #        "北环大道", 
    #        "桂庙路",
    #        "望海路", 
    #        "东滨路", 
    #        "内环路", 
    #        "兴海大道", 
    #        "南水路", 
    #        "港湾大道", 
    #        "赤湾二号路",
    #        ]
    # '新洲路','益田路'
    # lst = ['金田路']
    # err = []
    # for i in lst:
    #     try:
    #         print(i)
    #         main(i)
    #     except:
    #         err.append(i)
            
    # print(err)
    # main("广深沿江高速")
    # main("桂庙路")
    # main("兴海大道")

    pass

#%%
