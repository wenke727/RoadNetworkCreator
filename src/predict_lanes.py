#%%
import io
import copy
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

from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir, traverse_panos_by_rid, PANO_log
from road_network import OSM_road_network
from db.features import get_features
from db.db_process import load_from_DB, store_to_DB, update_lane_num_in_DB
from utils.geo_plot_helper import map_visualize
from utils.utils import load_config
from utils.img_process import get_pano_id_by_rid, plt_2_Image, cv2_2_Image, combine_imgs
from utils.classes import Digraph
from utils.spatialAnalysis import create_polygon_by_bbox, linestring_length
from model.lstr import draw_pred_lanes_on_img, lstr_pred

from road_matching import *

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)

config = load_config()
LSTR_DEBUG = config['data']['tmp']

df_memo = pd.read_csv(config['data']['df_pred_memo'])
df_memo.loc[:, 'pred'] = df_memo.pred.apply( lambda x: eval(x) )


# BBOX = [113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
# BBOX = [113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
BBOX = [113.92131,22.5235, 113.95630,22.56855] # 科技园片区
# BBOX = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区

#%%

VISITED = set()
ROAD_PANO_COUNT_DICT = {}


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

###
# ! SQL related
def query_df(df, att, val):
    val = '\''+val+'\'' if isinstance(val, str) else val 
    return df.query( f" {att} == {val} " )


# useless
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


def obtain_create_time(folder):
    import time
    fns = pd.DataFrame( [os.path.join(folder, f ) for f in os.listdir( folder )], columns=['file'] )
    fns.loc[:, 'mtime'] = fns.file.apply( lambda x: os.stat('./baidu_map.py').st_mtime )
    fns.loc[:, 'ctime'] = fns.file.apply( lambda x: os.stat('./baidu_map.py').st_ctime )
    
    return fns


# related
def get_panos_imgs_by_bbox(bbox=[113.92348,22.57034, 113.94372,22.5855], vis=True, with_folder=False):
    """获取某一个区域所有的panos

    Args:
        bbox (list, optional): [description]. Defaults to [113.92348,22.57034, 113.94372,22.5855].
        vis (bool, optional): [description]. Defaults to True.
        with_folder (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    res    = []
    
    # features = get_features('line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])
    features = get_features('line', bbox=bbox)
    if vis: map_visualize(features)

    for rid in tqdm(features.RID.unique(), desc='get_panos_imgs_by_bbox'):
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
        df_memo ([df]): prediction memoization.

    Returns:
        [type]: [description]
    """
    img_name = img_fn.split('/')[-1]

    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        info = lstr_pred( img_name )
        info['DIR'] = info['heading']
        df_memo = df_memo.append( info, ignore_index=True )

        return info, df_memo

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
    pred_img = cv2_2_Image(draw_pred_lanes_on_img( pred, None, dot=True, thickness=6, alpha=0.7, debug_infos=debug_infos) )
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


def lane_shape_predict_for_segment(rid, df_memo=df_memo, with_location=True, format='combine', folder = '../log', duration=0.5, gdf_road=None, 
                                   road_name=None, all_panos=False, quality=90):
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
    global VISITED, ROAD_PANO_COUNT_DICT
    if folder is not None and not os.path.exists(folder):  os.mkdir(folder)
    
    lst, df_pids = traverse_panos_by_rid(rid, DB_panos, PANO_log, all=all_panos)
    imgs = []

    for i in [ f.split("/")[-1] for f in lst]:
        index = None
        if VISITED is not None and ROAD_PANO_COUNT_DICT is not None:
            if i in VISITED: continue
            VISITED.add(i)
            ROAD_PANO_COUNT_DICT[road_name] = ROAD_PANO_COUNT_DICT.get( road_name, [] )
            index = len(ROAD_PANO_COUNT_DICT[road_name])
            ROAD_PANO_COUNT_DICT[road_name].append( (rid, i) )
        
        pred, df_memo = lane_shape_predict(i, df_memo)
            
        if with_location:
            img, img_fn = add_location_view_to_img( pred, whole_road=gdf_road, folder=folder, fn_pre=road_name, 
                                                   debug_infos=[] if index is None else [str(index)], quality=quality )
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
    pano_lst = pd.DataFrame({'name': pano_lst})
    unpredict_lst = pano_lst[pano_lst.merge(df_memo, how='left', on='name').lane_num.isna()]


    for i in tqdm( unpredict_lst.name.values):
        _, df_memo = lane_shape_predict(i, df_memo)
        

    pano_lst = pano_lst.merge(df_memo, how='left', on='name')
    
    return pano_lst, df_memo


def pred_osm_road_by_rid(road_id, roads_of_intrest, combineImgs=False, quality=100):
    """匹配osm某一条道路的所有panos，并预测

    Args:
        road_id (int, optional): [description]. Defaults to 633620767.
        roads_of_intrest ([type], optional): [description]. Defaults to rois.

    Returns:
        [type]: [description]
    """
    roads = query_df( roads_of_intrest, 'rid', road_id )
    if roads.shape[0] == 0: return False
    
    r = roads.iloc[0]
    road_level, road_name = r['road_type'], r['name']
    folder = LSTR_DEBUG + "/" + str(road_level)
    
    # if road_name is not None or road_name != '':
        # folder = LSTR_DEBUG + "/" + "_".join(  [str(road_level), str(road_name)] )
    # else:
        # folder = LSTR_DEBUG + "/" + "_".join(  [str(road_level), str(road_id)] )
        
    matching  = get_panos_of_road_by_id(road_id, roads_of_intrest, False)
    if matching is None or matching.shape[0] == 0:
        return []
    
    gdf_road = roads_of_intrest.query( f"rid=={road_id}" )
    fns = []
    for rid in matching.RID.values:
        format = 'combine'
        lane_shape_predict_for_segment(rid, df_memo, with_location=True, format=format, road_name=f"{road_id}_{road_name}",
                                       folder=folder, gdf_road=gdf_road, all_panos=False, quality=quality)
        # FIXME
        fns.append( folder+f"/{rid}.jpg" )

    if combineImgs:  
        combine = combine_imgs(fns)

        try:
            combine.save(LSTR_DEBUG+f"/{road_id}_combine.jpg",  "JPEG", quality=100, optimize=True, progressive=True)
        except IOError:
            # FIXME: write big picture
            combine = combine_imgs(fns[:20])
            PIL.ImageFile.MAXBLOCK = int(combine.size[0] * combine.size[1] * 2)
            combine.resize( (int(i/2) for i in combine.size) )
            combine.save(LSTR_DEBUG+f"/{road_id}_combine.jpg", "JPEG", quality=70, optimize=True)
        plt.close()

    return fns


def pred_analysis(BBOX):
    # TODO 对数据进行分析 离散性计算; 2 去除无用的数据，即非道路数据

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


def merge_rodas_segment(roads):
    net = Digraph( roads[['s','e']].values )

    net.combine_edges()
    result = net.combine_edges(roads)

    ids_sorted = []
    for _, road_ids in result:
        ids_sorted += road_ids
    
    return ids_sorted


def lstr_pred_by_bbox(BBOX):
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




#%%

if __name__ == "__main__":
    import pickle
    BBOX = [113.92131,22.5235, 113.95630,22.56855] # 科技园片区
    lstr_pred_by_bbox(BBOX)
    BBOX = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
    lstr_pred_by_bbox(BBOX)

    try:
        pickle.dump(VISITED, open('./log/VISITED.pkl', 'wb'))
        pickle.dump(ROAD_PANO_COUNT_DICT, open('./log/ROAD_PANO_COUNT_DICT.pkl', 'wb'))
    except:
        pass

    # 获取某一道路反方向的大图
    tmp = _get_revert_df_edges(-208128052, df_edges)
    pred_osm_road_by_rid(-208128052, tmp, True)

    pred_osm_road_by_rid(208128052, df_edges, True)


#%%


# # 合并成一张大图
# road_id = 243387686
# roads_of_intrest = df_edges.copy()
# combineImgs = True
# pred_osm_road_by_rid(243387686, df_edges, True)
# pred_osm_road_by_rid(529070115, df_edges, True)
# pred_osm_road_by_rid(208128052, df_edges, True)










# test = query_df(df_edges, 'rid', 208128052)

# map_visualize(test)

# from utils.collection.roadNetwork import reverse_shp
# test.loc[:, 'geometry'] = test.geometry.apply( lambda x: reverse_shp(x) )

# # ! 尝试反转 道路，然后去匹配反方向的道路, 遍历后的顺序问题，解决一下
# roads = test
# roads.loc[:, 'rid'] = -roads.rid



# df_edges = df_edges.append( roads, ignore_index=True )







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


    # lane_shape_predict_for_segment('52bb2b-4ac7-dc4b-206c-a3ccfe', df_memo, True, 'combine', LSTR_DEBUG, gdf_road=df_edges.query( "rid==633620767" ))
    # # 尝试多线程，但感觉仅仅是单线程在启动
    # from utils.parrallel_helper import apply_parallel, apply_parallel_helper
    # apply_parallel_helper(lane_shape_predict_for_segment, res.head(3), "RID", folder=LSTR_DEBUG)

    # pred_osm_road_by_rid(557225633)
    # query_df(df_memo, "RID", '97571c-f6a6-16e9-6440-54d7e1')
    # lane_shape_predict_for_segment('97571c-f6a6-16e9-6440-54d7e1', df_memo, True, 'combine', folder, gdf_road=gdf_road, all_panos=False)
        