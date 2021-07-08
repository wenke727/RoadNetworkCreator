#%%
import io
import os, sys
import math
import shutil
import pandas as pd
import geopandas as gpd
import PIL
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
from tqdm import tqdm

from pano_img import traverse_panos_by_rid, PANO_log
from road_network import OSM_road_network
from road_matching import * # df_edges: osm_shenzhen.edges

from db.features_API import get_features
from utils.utils import load_config
from utils.classes import Digraph
from utils.df_helper import query_df, load_df_memo
from utils.geo_plot_helper import map_visualize
from utils.img_process import plt_2_Image, cv2_2_Image, combine_imgs
from utils.spatialAnalysis import create_polygon_by_bbox, linestring_length
from model.lstr import draw_pred_lanes_on_img, lstr_pred, lstr_pred_by_pid

# DB_panos, DB_roads = load_DB_panos(), load_DB_roads()
df_edges = get_features('edge')
DB_panos = get_features('point')
DB_roads = get_features('line')

config = load_config()
LSTR_DEBUG_FOLDER = config['data']['tmp']
df_memo = load_df_memo(config['data']['df_pred_memo'])
VISITED = set()
ROAD_PANO_COUNT_DICT = {}

#%%

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
        from road_matching import get_panos_of_road_and_indentify_lane_type_by_id
        matching0 = get_panos_of_road_and_indentify_lane_type_by_id(-road_id, df_tmp, False)
        matching1 = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False)
        _, ax = map_visualize(matching0, scale =0.001)
        matching1.plot(column='level_0', legend=True, ax=ax, cmap='jet')
        matching0.plot(column='level_0', legend=True, ax=ax, cmap='jet')

    return df_tmp


def merge_rodas_segment(roads):
    net = Digraph( roads[['s','e']].values )

    net.combine_edges()
    result = net.combine_edges(roads)

    ids_sorted = []
    for _, road_ids in result:
        ids_sorted += road_ids
    
    return ids_sorted


def get_panos_imgs_by_bbox(bbox=[113.92348,22.57034, 113.94372,22.5855], vis=True, with_folder=False):
    """给定一个区域，获取所有的panos

    Args:
        bbox (list, optional): [description]. Defaults to [113.92348,22.57034, 113.94372,22.5855].
        vis (bool, optional): [description]. Defaults to True.
        with_folder (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    res = []
    features = get_features('line', bbox=bbox)
    if vis: map_visualize(features)

    for rid in tqdm(features.RID.unique(), desc='get_panos_imgs_by_bbox'):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log, all=False)
        res += info
    print( 'total number of pano imgs: ', len(res))
    
    # folder = './images'
    # dst    = "~/Data/TuSimple/LSTR/lxd"    
    if False:
        if not os.path.exists(folder): os.mkdir(folder)
        for fn in res: shutil.copy( fn, folder )
        cmd = os.popen( f" mv {folder} {dst} " ).read()

    if not with_folder:
        res = [ i.split('/')[-1] for  i in res]
    
    return res


def lane_shape_predict(img_fn, df_memo):
    """针对img_fn的照片进行车道线形预测，并将结果缓存，便于再次访问

    Args:
        img_fn ([type]): [description]
        df_memo ([df]): prediction memoization.

    Returns:
        [type]: [description]
    """
    img_name = img_fn.split('/')[-1]

    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        print(img_name)
        info = lstr_pred( img_name )
        info['DIR'] = info['heading']
        df_memo = df_memo.append( info, ignore_index=True )

        return info, df_memo

    return df_memo.query( f" name == '{img_name}' " ).to_dict('records')[0], df_memo


def update_unpredict_panos(pano_lst=DB_panos.PID.unique(), df_memo=df_memo):
    """更新区域内还没有预测的pano照片数据

    Args:
        pano_lst ([type]): [description]
        df_memo ([type]): [description]

    Returns:
        [type]: [description]
    """
    pano_lst = pd.DataFrame({'name': pano_lst})
    unpredict_lst = pano_lst[pano_lst.merge(df_memo, how='left', left_on='name', right_on='PID').lane_num.isna()]
    unpredict_lst = unpredict_lst.merge(DB_panos, left_on='name', right_on='PID')
    queue = unpredict_lst.apply(lambda x: f"{x.RID}_{x.Order:02d}_{x.PID}_{x.DIR}.jpg", axis=1)
    
    for i in tqdm( queue.values, 'update_unpredict_panos'):
        _, df_memo = lane_shape_predict(i, df_memo)
    pano_lst = pano_lst.merge(df_memo, how='left', on='name')
    
    # TODO: save it to csv, but there is a confict with `/Data/LaneDetection_PCL/LSTR/predict.py.predict_bacth`
    
    return pano_lst, df_memo


# 标注相关
def lstr_pred_check_by_bbox(BBOX):
    """预测区域内的街景，并根据道路等级保存在子文件夹中，便于后续的标注工作的筛选

    Args:
        BBOX ([type]): [description]

    Returns:
        [type]: [description]
    """
    rois = gpd.clip(df_edges, create_polygon_by_bbox(BBOX) )
    linestring_length(rois, 'length')
    rois.loc[rois.name.isna(), 'name'] = rois.loc[rois.name.isna()].road_type.apply( lambda x:  "".join([ i[0] for i in  x.split("_")]))
    sorted(rois.road_type.unique())

    road_type_lst =  [
        'motorway',
        'trunk', 
        'primary',
        'secondary',
        'tertiary',
        'motorway_link',
        'trunk_link',
        'primary_link',
        'secondary_link',
        'residential',
        'unclassified'
        'construction',
    ]

    error_lst = ['error list']
    for road_type in road_type_lst[:]:
        roads = rois.query( f"road_type == '{road_type}' " ) 
        # map_visualize(roads)

        for road_name, df in roads.groupby('name'):
            # if road_name !='_': continue
            rids = merge_rodas_segment(df)

            for i in rids[:]:
                try:
                    pred_osm_road_by_rid(i, rois)
                except:
                    error_lst.append( f"{road_type}, {road_name}, {i}" )

    return error_lst


def lstr_pred_check_for_label():
    """用现有算法预测框选区域的街景，后续将挑选错误的标注，并重新标注
    """
    import pickle
    BBOX = [113.92131,22.5235, 113.95630,22.56855] # 科技园片区
    lstr_pred_check_by_bbox(BBOX)
    BBOX = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
    lstr_pred_check_by_bbox(BBOX)

    try:
        pickle.dump(VISITED, open('./log/VISITED.pkl', 'wb'))
        pickle.dump(ROAD_PANO_COUNT_DICT, open('./log/ROAD_PANO_COUNT_DICT.pkl', 'wb'))
    except:
        pass

    return 


# 绘图相关
def plot_pano_and_its_view(pid, DB_panos, DB_roads, road=None,  heading=None):
    """绘制pano位置图，包括所在的路段，位置以及视角

    Args:
        pid ([type]): Pano id
        DB_panos ([type]): Panos DB.
        DB_roads ([type]): Roads DB.
        road ([type], optional): The whole road need to plot in the figure. Defaults to None.
        heading ([type], optional): The view heading. Defaults to None.

    Returns:
        [Image]: Image with the predicted info.
    """

    rid = DB_panos.query( f"PID=='{pid}' " ).RID.iloc[0]
    pid_record = query_df(DB_panos, "RID", rid).query( f"PID == '{pid}'" )
    assert( len(pid_record) > 0 )
    pid_record = pid_record.iloc[0]

    if heading is None:
        heading = pid_record.DIR
    x, y = pid_record.geometry.coords[0]
    
    if road is None:
        fig, ax = map_visualize( DB_roads.query( f"RID == '{pid_record.RID}' " ), label="Lane" )
    else:
        fig, ax = map_visualize( DB_roads.query( f"RID == '{pid_record.RID}' " ).append(road), color='blue', linestyle='--')
        road.plot(ax=ax, label="Road", color='red')
        # fig, ax = map_visualize( road, label="Road" )

    x0, x1 = ax.get_xlim()
    aus_line_len = (x1-x0)/20
    dy, dx = math.cos(heading/180*math.pi) * aus_line_len, math.sin(heading/180*math.pi) * aus_line_len
    
    ax.annotate('', xy=(x+dx, y+dy), xytext= (x,y) ,arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.5))
    gpd.GeoSeries( [Point(x, y)] ).plot(ax=ax, label='Pano', marker='*',  markersize= 360 )

    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.close()
    
    return fig


def add_location_view_to_img(pred, whole_road=None, debug_infos=[], folder='../log', fn_pre=None, width=None, quality=90):
    """在预测的街景照片中添加位置照片

    Args:
        pred ([type]): [description]
        folder (str, optional): [description]. Defaults to '../log'.

    Returns:
        [type]: [description]
    """
    pre = f"{int(debug_infos[0]):03d}_" if len(debug_infos) > 0 else ""

    fig = plot_pano_and_its_view( pred['PID'], DB_panos, DB_roads, whole_road )
    loc_img = plt_2_Image(fig)
    debug_infos += [pred['RID'], pred['PID']]
    pred_img = draw_pred_lanes_on_img( pred, None, dot=True, thickness=6, alpha=0.7, debug_infos=debug_infos) 
    plt.close()

    w0, h = pred_img.size
    f = loc_img.size[1] / h
    w = int(loc_img.size[0]/f)

    to_img = Image.new('RGB', ( pred_img.size[0] + w, pred_img.size[1]))
    to_img.paste( pred_img, (0, 0) )
    to_img.paste( loc_img.resize( (w, h), Image.ANTIALIAS), (w0, 0) )
    
    if width is not None:
        to_img = to_img.resize( (width, int(to_img.size[1] / to_img.size[0]*width)) )
    
    if folder is not None:
        # img_fn =  (fn_pre + "_" if fn_pre is not None else '') + pre + f"{pred['name'].replace('.jpg', '_loc.jpg')}"
        img_fn =  (fn_pre + "_" if fn_pre is not None else '') + pre + pred['name']
        img_fn = os.path.join( folder, img_fn )
        to_img.save( img_fn, quality=quality)

        return to_img, img_fn

    return to_img, None


def lane_shape_predict_for_rid_segment(rid, 
                                       df_memo=df_memo, 
                                       with_location=True, 
                                       format='combine', 
                                       folder = '../log', 
                                       duration=0.5, 
                                       gdf_road=None,
                                       road_name=None, 
                                       all_panos=False, 
                                       quality=90, 
                                       verbose=False):
    """预测某一个路段所有的街景，并生成预测动画或图片

    Args:
        rid ([type]): [description]
        df_memo ([type]): [description]
        with_location (bool, optional): [description]. Defaults to True.
        format (str, optional): [description]. [single, combine, gif] Defaults to `combine`.
        folder (str, optional): [description]. Defaults to '../log'.
        duration (float, optional): [description]. Defaults to 0.33.

    Returns:
        [type]: [description]
    """
    # global VISITED, ROAD_PANO_COUNT_DICT
    if folder is not None and not os.path.exists(folder):
        os.mkdir(folder)
    
    lst, df_pids = traverse_panos_by_rid(rid, DB_panos, PANO_log, all=all_panos)
    imgs = []

    for i in [ f.split("/")[-1] for f in lst]:
        index = None
        # if VISITED is not None and ROAD_PANO_COUNT_DICT is not None:
            # if i in VISITED: continue
            # VISITED.add(i)
            # ROAD_PANO_COUNT_DICT[road_name] = ROAD_PANO_COUNT_DICT.get( road_name, [] )
            # index = len(ROAD_PANO_COUNT_DICT[road_name])
            # ROAD_PANO_COUNT_DICT[road_name].append( (rid, i) )
        
        pred, df_memo = lane_shape_predict(i, df_memo)
            
        if with_location:
            img, img_fn = add_location_view_to_img(pred, 
                                                   whole_road=gdf_road, 
                                                   folder=folder, 
                                                   fn_pre=road_name, 
                                                   debug_infos=[] if index is None else [str(index)], 
                                                   quality=quality )
        else:
            img_fn = os.path.join(folder, pred['name'])
            img = draw_pred_lanes_on_img( pred, img_fn )

        imgs.append(img)
    
    fn = None
    if format=='combine':
        all_imgs = combine_imgs(imgs)
        rid = df_pids.iloc[0].RID
        fn = os.path.join(folder, rid+".jpg")
        all_imgs.save(fn, quality=100)

    if format=='gif':
        import imageio
        images = []
        for img in imgs:
            # images.append(imageio.imread(filename))
            images.append(img)

        fn = os.path.join(folder, rid+".gif")
        imageio.mimsave(fn, images, 'GIF', duration=duration)
    
    return fn


# check
def pred_osm_road_by_rid(road_id, roads_of_intrest, combineImgs=False, quality=100):
    """基于osm中rid的道路匹配panos，然后调用模型预测车道线情况，将预测结果绘制在街景上

    Args:
        road_id (int, optional): Road id in the OSM xml file.
        roads_of_intrest (pd.DataFrame): The road features crawled from OSM and preprocess in `road_matching.py` script.

    Returns:
        [type]: [description]
    """
    roads = query_df( roads_of_intrest, 'rid', road_id )
    if roads.shape[0] == 0: 
        return None
    
    r = roads.iloc[0]
    road_level, road_name = r['road_type'], r['name']
    folder = LSTR_DEBUG_FOLDER + "/" + str(road_level)
    
    matching  = get_panos_of_road_by_id(road_id, roads_of_intrest, False)
    if matching is None or matching.shape[0] == 0:
        return []
    
    gdf_road = roads_of_intrest.query( f"rid=={road_id}" )

    fns = []
    rids_lst_inorder = []
    for rid in matching.RID.values:
        if rid in rids_lst_inorder:
            continue
        rids_lst_inorder.append(rid)
        format = 'combine' if combineImgs else None
        fn = lane_shape_predict_for_rid_segment(rid, 
                                           df_memo, 
                                           with_location=True, 
                                           format=format, 
                                           road_name=f"{road_id}_{road_name}",
                                           folder=folder, 
                                           gdf_road=gdf_road, 
                                           all_panos=False, 
                                           quality=quality)
        fns.append( fn )

    if combineImgs:  
        combine = combine_imgs(fns)

        try:
            combine.save(LSTR_DEBUG_FOLDER+f"/{road_id}_combine.jpg",  "JPEG", quality=100, optimize=True, progressive=True)
        except IOError:
            # FIXME: write big picture
            combine = combine_imgs(fns[:20])
            PIL.ImageFile.MAXBLOCK = int(combine.size[0] * combine.size[1] * 2)
            combine.resize( (int(i/2) for i in combine.size) )
            combine.save(LSTR_DEBUG_FOLDER+f"/{road_id}_combine.jpg", "JPEG", quality=70, optimize=True)
        plt.close()

    return fns, rids_lst_inorder


def pred_analysis(BBOX, df_memo=df_memo):
    pano_lst, df_memo = update_unpredict_panos(get_panos_imgs_by_bbox(BBOX), df_memo)
    pano_lst[['RID', 'lane_num']].groupby('RID').count().query('lane_num > 1')

    def count_pred_num(df):
        return df.lane_num.unique()

    groups = pano_lst[['RID', 'lane_num']].groupby('RID')
    res = pd.DataFrame(groups.apply( count_pred_num ), columns=['pred_sit'])
    res.loc[:,'std'] = groups.std().fillna(-1).values
    res.loc[:,'count'] = groups.count().values
    res.reset_index(inplace=True)

    # TODO: add length, caculate the heading
    res = res.merge(
            DB_panos[['RID','Order']].groupby('RID').count().reset_index()
        ).rename(columns={'Order': "panos_num"})

    return res


#%%
if __name__ == "__main__":
    """用现有算法预测框选区域的街景，后续将挑选错误的标注"""
    BBOX = [113.92131,22.5235, 113.95630,22.56855] # 科技园片区
    lstr_pred_check_for_label()


    """ 预测没有预测的结果, 并更新数据库 """
    update_unpredict_panos()
    # from db.db_process import update_lane_num_in_DB
    # update_lane_num_in_DB()


    """ 对百度地图的 路段 进行可视化 """
    rid = '3469fc-471c-d504-ab47-1a3284'
    lane_shape_predict_for_rid_segment(rid, df_memo, True, 'combine', LSTR_DEBUG_FOLDER)
    lane_shape_predict_for_rid_segment('3e9933-e732-1db9-61dc-ee8b54', df_memo, True, 'gif', LSTR_DEBUG_FOLDER, gdf_road=df_edges.query( "rid==633620767" ))


    """ 预测 OSM一条道路 的情况 """
    fns, rids_lst = pred_osm_road_by_rid(208128052, df_edges, combineImgs=True)
    # 预测其反方向情况
    tmp = _get_revert_df_edges(-208128052, df_edges)
    pred_osm_road_by_rid(-208128052, tmp, combineImgs=True)


    # BUG 一条路仅有一个点的情况下，需要更新DIR数值； 终点也需要更新数值
    # # module: update the heading of the last point 
    # df_road_with_1_node = DB_panos[['RID', 'PID']].groupby("RID").count().rename(columns={"PID":'count'}).reset_index()
    # df_road_with_1_node.query("count==1", inplace=True)

    # dirs = df_road_with_1_node.apply( lambda x: get_heading_according_to_prev_road(x.RID), axis=1 )
    # df_road_with_1_node.loc[:, 'dirs'] = dirs

    # df_road_with_1_node.to_csv("../df_road_with_1_node.csv")

