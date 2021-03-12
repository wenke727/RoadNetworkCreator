#%%
import io
import copy
import os, sys
import math
import shutil
import pandas as pd
import geopandas as gpd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir, traverse_panos_by_rid, PANO_log
from road_network import OSM_road_network
from db.features import get_features
from db.db_process import load_from_DB, store_to_DB
from utils.geo_plot_helper import map_visualize
from utils.utils import load_config
from utils.img_process import get_pano_id_by_rid, plot_pano_and_its_view, plt_2_Image, cv2_2_Image
from model.lstr import draw_pred_lanes_on_img, lstr_pred


config = load_config()
DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)

LSTR_DEBUG = "../../0_tmp"

df_memo = pd.read_csv(config['data']['df_pred_memo'])
df_memo.loc[:, 'pred'] = df_memo.pred.apply( lambda x: eval(x) )


# BBOX = [113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
BBOX = [113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
# BBOX = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区

#%%

###
# ! SQL related
def query_df_memo_by_rid(rid = '0019c9-7503-4f2e-3f59-bbbccf'):
    return df_memo.query( f"RID == '{rid}' " ).sort_values(by='name')

def query_by_rid(df, rid = '0019c9-7503-4f2e-3f59-bbbccf'):
    return df.query( f"RID == '{rid}' " )

###


def get_heading_according_to_prev_road(rid):
    """ 可能会有多个值返回，如：
        rid = 'ba988a-7763-e9af-a5fb-dc8590'
    """
    # initiale:
    # from db.db_process import load_from_DB, extract_connectors_from_panos_respond
    # connecters = extract_connectors_from_panos_respond( DB_pano_base, DB_roads )
    # DB_connectors = connecters
    # store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    # from utils.utils import load_config
    # from sqlalchemy import create_engine
    # import geopandas as gpd
    # import pandas as pd

    # config   = load_config()
    # ENGINE   = create_engine(config['data']['DB'])
   
    sql = f"""SELECT panos.* FROM 
            (SELECT "RID", max("Order") as "Order" FROM
                (
                SELECT * FROM panos 
                WHERE "RID" in
                    (
                        SELECT "RID" FROM panos 
                        WHERE "PID" in 
                        (
                            SELECT prev_pano_id FROM connectors 
                            WHERE "RID" = '{rid}'
                        )
                    )
                ) a
            group by "RID") b,
            panos
        WHERE panos."RID" = b."RID" and panos."Order" = b."Order"
        """
    res = pd.read_sql( sql, con=ENGINE )
    res

    return res.DIR.values.tolist()


def images_to_video(path):
    import cv2
    input_folder = "./detections"

    fns = os.listdir(input_folder)

    img = cv2.imread( os.path.join(input_folder, fns[0]) )
    x, y, z = img.shape
    img_array = []
 
    for f in fns:
        img = cv2.imread(os.path.join(input_folder, f))
        if img is None:
            print(f + " is error!")
            continue
        img_array.append(img)
    
    len(img_array)
    # 图片的大小需要一致
    # img_array, size = resize(img_array, 'largest')
    
    fps = 1
    out = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (x,y ), True)
 
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()


def calc_angle(item): 
    """计算GPS坐标点的方位角 Azimuth 

    Args:
        item ([type]): [description]

    Returns:
        [type]: [description]
    """
    angle=0
    
    x1, y1 = item.p0[0], item.p0[1]
    x2, y2 = item.p1[0], item.p1[1]
    
    dy= y2-y1
    dx= x2-x1
    if dx==0 and dy>0:
        angle = 0
    if dx==0 and dy<0:
        angle = 180
    if dy==0 and dx>0:
        angle = 90
    if dy==0 and dx<0:
        angle = 270
    if dx>0 and dy>0:
       angle = math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy>0:
       angle = 360 + math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    elif dx>0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    return angle


def calc_angle_for_df(df):
    coords = df.geometry.apply(lambda x: x.coords[0]) 
    df_new = pd.DataFrame()
    df_new['p0'], df_new['p1'] = coords, coords.shift(-1)

    df_new[:-1].apply(lambda x: calc_angle(x), axis=1)


def draw_network_lanes( fn = "../lxd_predict.csv", save_img=None ):
    """draw High-precision road network 

    Args:
        fn (str, optional): [description]. Defaults to "../lxd_predict.csv".
        save_img ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    from scipy import stats
    colors = ['black', 'blue', 'orange', 'yellow', 'red']
    widths = [0.75, 0.9, 1.2, 1.5, 2.5, 3]
    
    df = pd.read_csv(fn)
    df.loc[:, "RID"] = df.name.apply( lambda x: x.split('/')[-1].split('_')[-4] )
    df = df.groupby( 'RID' )[['lane_num']].agg( lambda x: stats.mode(x)[0][0] ).reset_index()
    df.loc[:, 'lane_num'] = df.lane_num - 1

    matching = DB_roads.merge( df, on = 'RID' )
    max_lane_num = matching.lane_num.max()

    fig, ax = map_visualize(matching, color='gray', scale=.05, figsize=(12, 12))
    for i in range(max_lane_num):
        matching.query(f'lane_num=={i+1}').plot(color = colors[i], linewidth = widths[i], label =f'{i+1} lanes', ax=ax)
    plt.legend()
    plt.close()
    
    if save_img is not None: plt.savefig(save_img, dpi =500)

    return matching


def get_panos_imgs_by_bbox(bbox=[113.92348,22.57034, 113.94372,22.5855], vis=True, with_folder=False):
    # 获取某一个区域所有的panos
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    res    = []
    
    # features = get_features('line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])
    features = get_features('line', bbox=bbox)
    if vis: map_visualize(features)

    for rid in tqdm(features.RID.unique()):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log, all=False)
        res += info

    print( 'total number of pano imgs: ', len(res))
    
    if False:
        if not os.path.exists(folder): os.mkdir(folder)

        for fn in res: shutil.copy( fn, folder )
        cmd = os.popen( f" mv {folder} {dst} " ).read()

    if not with_folder:
        res = [ i.split('/')[-1] for  i in res]
    
    return res


def lane_shape_predict(img_fn, df_memo):
    """车道线形预测

    Args:
        img_fn ([type]): [description]
        df_memo ([type]): [description]

    Returns:
        [type]: [description]
    """
    img_name = img_fn.split('/')[-1]

    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        print(f"query data: {img_name}")
        info = lstr_pred( img_name )
        info['respond']['PID'] = img_name.split('_')[2]
        info['respond']['DIR'] = int(img_name.split('_')[-1].split('.')[0])
        df_memo = df_memo.append( info['respond'], ignore_index=True )

        return info['respond'], df_memo

    return df_memo.query( f" name == '{img_name}' " ).to_dict('records')[0], df_memo


def plot_pano_and_its_view(pid, DB_panos, DB_roads, road=None,  heading=None):
    """绘制pano所在的路段，位置以及视角

    Args:
        pid ([type]): [description]
        road (gpd): the whole road
    """
    rid = DB_panos.query( f"PID=='{pid}' " ).RID.iloc[0]
    pid_record = get_pano_id_by_rid(rid, DB_panos).query( f"PID == '{pid}'" )
    assert( len(pid_record) > 0 )
    pid_record = pid_record.iloc[0]

    if heading is None:
        heading = pid_record.DIR
    x, y = pid_record.geometry.coords[0]
    
    if road is None:
        fig, ax = map_visualize( DB_roads.query( f"RID == '{pid_record.RID}' " ), label="Lane" )
    else:
        fig, ax = map_visualize( road, label="Road" )

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


def add_location_view_to_img(pred, whole_road=None, folder='../log'):
    """在预测的街景照片中添加位置照片

    Args:
        pred ([type]): [description]
        folder (str, optional): [description]. Defaults to '../log'.

    Returns:
        [type]: [description]
    """
    # fig, ax = map_visualize(DB_roads.query(f" RID == '{pred['RID']}' "), figsize=(12,12))
    # DB_panos.query(f" PID == '{pred['PID']}'").plot(ax=ax)
    # plt.tight_layout(pad=0.001)
    # TODO 增加全局道路的图片
    fig = plot_pano_and_its_view( pred['PID'], DB_panos, DB_roads, whole_road )
    loc_img = plt_2_Image(fig)
    
    # pred_img = cv2_2_Image(draw_pred_lanes_on_img( pred, None))
    pred_img = cv2_2_Image(draw_pred_lanes_on_img( pred, None, dot=True, thickness=6, alpha=0.7, debug_infos=[pred['RID'], pred['PID']]) )
    plt.close()

    # pred_img = Image.open(pred['name'])
    # loc_img = Image.open('test.jpg')

    w0, h = pred_img.size
    f = loc_img.size[1] / h
    w = int(loc_img.size[0]/f)

    to_img = Image.new('RGB', ( pred_img.size[0] + w, pred_img.size[1]))
    to_img.paste( pred_img, (0, 0) )
    to_img.paste( loc_img.resize( (w, h), Image.ANTIALIAS), (w0, 0) )
    
    to_img = to_img.resize( (1024, int(to_img.size[1] / to_img.size[0]*1024)) )
    
    if folder is not None:
        img_fn = os.path.join( folder, pred['name'].replace('.jpg', '_loc.jpg') )
        to_img.save( img_fn, quality=90)
    
        return to_img, img_fn

    return to_img, None


def combine_imgs(imgs):
    n = len(imgs)
    if n == 0: return None
    
    if not isinstance(imgs[0], Image.Image):
        imgs = [ Image.open(i) for i in imgs ]
    
    heights = [ img.size[1] for img in imgs ]
    
    w, _ = imgs[0].size
    to_img = Image.new('RGB', ( w, sum(heights)))

    acc_h = 0
    for index, i in enumerate(imgs):
        to_img.paste( i, (0, acc_h ) )
        acc_h += heights[index]

    return to_img


def lane_shape_predict_for_segment(rid, df_memo=df_memo, with_location=True, format='combine', folder = '../log', duration=0.5, gdf_road=None, all_panos=False):
    """[预测某一个路段所有的街景，并生成预测动画]

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
    if folder is not None and not os.path.exists(folder): os.mkdir(folder)
    lst, df_pids = traverse_panos_by_rid(rid, DB_panos, PANO_log, all=all_panos)

    imgs = []
    for i in tqdm(lst, desc="lane_shape_predict_for_segment: "):
        pred, df_memo = lane_shape_predict(i, df_memo)
        if with_location:
            # TODO 针对一条线路的情况
            img, img_fn = add_location_view_to_img( pred, whole_road=gdf_road, folder=None )
        else:
            img_fn = os.path.join(folder, pred['name'])
            img = draw_pred_lanes_on_img( pred, img_fn )
        imgs.append(img)

    if format=='combine':
        all_imgs = combine_imgs(imgs)
        rid = df_pids.iloc[0].RID
        all_imgs.save( os.path.join(folder, rid+".jpg"), quality=100)

    if format=='gif':
        import imageio
        images = []
        for img in imgs:
            # print(filename)
            # images.append(imageio.imread(filename))
            images.append(img)
        imageio.mimsave(f'{folder}/{rid}.gif', images, 'GIF', duration=duration)
    
    return True


def update_unpredict_panos(pano_lst, df_memo):
    """更新区域内还没有预测的pano照片数据

    Args:
        pano_lst ([type]): [description]
        df_memo ([type]): [description]

    Returns:
        [type]: [description]
    """
    # df_memo.loc[:, 'RID'] = df_memo.name.apply(lambda x: x.split("_")[0])
    # df_memo.loc[:, 'name'] = df_memo.name.apply( lambda x: x.split('/')[-1] )
    # df_memo[['PID','RID', 'DIR', 'lane_num', 'name', 'pred']].to_csv(config['data']['df_pred_memo'], index=False)

    pano_lst = pd.DataFrame({'name': pano_lst})
    unpredict_lst = pano_lst[pano_lst.merge(df_memo, how='left', on='name').lane_num.isna()]


    for i in tqdm( unpredict_lst.name.values):
        _, df_memo = lane_shape_predict(i, df_memo)
        

    pano_lst = pano_lst.merge(df_memo, how='left', on='name')
    
    return pano_lst, df_memo



#%%

# if __name__ == '__main__':
#     main()
    
    # lane_shape_predict_for_segment('fbc78a-586c-f310-7326-6e9812', df_memo, True, True, LSTR_DEBUG)
    # lane_shape_predict_for_segment('0019c9-7503-4f2e-3f59-bbbccf', df_memo, True, True, LSTR_DEBUG)

    
    # img_fn = 'fbc78a-586c-f310-7326-6e9812_39_09005700011601081212474588N_342.jpg'
    # pred, df_memo = lane_shape_predict(img_fn, df_memo)
    # draw_pred_lanes_on_img( pred, None )
    # BUG 一条路仅有一个点的情况下，需要更新DIR数值； 终点也需要更新数值


    # # module: update the heading of the last point 
    # df_road_with_1_node = DB_panos[['RID', 'PID']].groupby("RID").count().rename(columns={"PID":'count'}).reset_index()
    # df_road_with_1_node.query("count==1", inplace=True)

    # dirs = df_road_with_1_node.apply( lambda x: get_heading_according_to_prev_road(x.RID), axis=1 )
    # df_road_with_1_node.loc[:, 'dirs'] = dirs

    # df_road_with_1_node.to_csv("../df_road_with_1_node.csv")
    
        
#%%

def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
    # ! modified
    """obtain the panos in road[rid] 

    Args:
        rid (str): the id of road segements

    Returns:
        [type]: [description]
    """
    
    df_pids = get_pano_ids_by_rid(rid, DB_panos)
    
    pano_lst = df_pids[['Order','PID', 'DIR']].values
    length = len(pano_lst)
    res, pre_heading = [], 0
    
    for id, (order, pid, heading) in enumerate(pano_lst):
        if heading == 0 and id != 0:   # direction, inertial navigation
            heading = pre_heading
        
        if not all:
            if length > 3 and order % 3 != 1:
                # print(order)
                continue

        fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        res.append(get_staticimage(pid=pid, heading=heading, path=fn, log_helper=log))
        pre_heading = heading
        
    return res, df_pids


#! 对数据进行分析 离散性计算，

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
res
res = res.merge(
        DB_panos[['RID','Order']].groupby('RID').count().reset_index()
    ).rename(columns={'Order': "panos_num"})



for num in res.panos_num.unique():
    rids = res.query( f"panos_num == {num}" ).RID.values
    for i in rids:
        lane_shape_predict_for_segment(i, df_memo, True, 'combine', LSTR_DEBUG+f"/{num}")



std = -1
rids = res.query( "-1==std" ).RID.values
for i in rids:
    lane_shape_predict_for_segment(i, df_memo, True, 'combine', LSTR_DEBUG+f"/std_-1")





lane_shape_predict_for_segment('52bb2b-4ac7-dc4b-206c-a3ccfe', df_memo, True, 'combine', LSTR_DEBUG, gdf_road=df_edges.query( "rid==633620767" ))
# # 尝试多线程，但感觉仅仅是单线程在启动
# from utils.parrallel_helper import apply_parallel, apply_parallel_helper
# apply_parallel_helper(lane_shape_predict_for_segment, res.head(3), "RID", folder=LSTR_DEBUG)



#%%

# from road_matching import *

def pred_osm_road_by_rid(road_id = 633620767):

    matching  = get_panos_of_road_by_id(road_id, df_edges, True)
    gdf_road = df_edges.query( f"rid=={road_id}" )

    fns = []
    for RID in matching.RID.values:
        lane_shape_predict_for_segment(RID, df_memo, True, 'combine', LSTR_DEBUG+f"/{road_id}", gdf_road=gdf_road, all_panos=False)
        fns.append( LSTR_DEBUG+f"/{road_id}/{RID}.jpg" )

    combine = combine_imgs(fns)
    combine.save(  LSTR_DEBUG+f"/{road_id}_combine.jpg", quality=100 )

    return combine


pred_osm_road_by_rid()


pred_osm_road_by_rid(633620762)
