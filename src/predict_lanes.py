import os, sys
import math
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from road_network import OSM_road_network

from db.features import get_features
from db.db_process import load_from_DB
from road_matching import traverse_panos_by_rid
from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir
from utils.geo_plot_helper import map_visualize
from pano_img import PANO_log
from utils.utils import load_config

config = load_config()
DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)

# road_names = ['打石一路', '创科路']
# roads = df_edges.query( f"name in {road_names} " )
# features.append( roads )
# map_visualize(features)


def images_to_video(path):
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


def url_to_image_CV2(url='http://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png', type=''):
    import numpy as np
    import cv2
    from urllib.request import urlopen

    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image


def draw_lanes( fn = "../lxd_predict.csv", save_img=None ):
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
    
    if save_img is not None: plt.savefig(save_img, dpi =500)

    return matching


def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
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


def get_panos_imgs_by_bbox():
    #! 获取某一个区域所有的panos
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    res    = []
    
    # features = get_features('line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])
    bbox=[113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
    # bbox=[113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
    features = get_features('line', bbox=bbox)
    map_visualize(features)

    for rid in tqdm(features.RID.unique()):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log)
        res += info
    len(res)
    
    if not os.path.exists(folder): os.mkdir(folder)

    for fn in res:
        shutil.copy( fn, folder )

    print( 'total number of pano imgs: ', len(res))

    cmd = os.popen( f" mv {folder} {dst} " ).read()

    return res

#%%

if __name__ == '__main__':
        
    points = get_features('point', bbox=[113.929807, 22.573710, 113.937680, 22.578734])
    map_visualize(points)


    df = draw_lanes( '../predict.csv' )


    df.to_file('../lxd_lanes.geojson', driver="GeoJSON")
    points.to_file('../lxd_points.geojson', driver="GeoJSON")

    
    df = draw_lanes( '../lxd_predict.csv' )








#%%
# 预测模块


from model.lstr import draw_pred_lanes_on_img, lstr_pred

def lane_shape_predict(img_fn, df_memo):
    img_name = img_fn.split('/')[-1]

    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        print(f"query data: {img_name}")
        info = lstr_pred( img_name )
        info['respond']['PID'] = img_name.split('_')[2]
        info['respond']['DIR'] = int(img_name.split('_')[-1].split('.')[0])
        df_memo = df_memo.append( info['respond'], ignore_index=True )

        return info['respond'], df_memo

    return df_memo.query( f" name == '{img_name}' " ).to_dict('records')[0], df_memo


def lane_shape_predict_for_segment(rid, df_memo, folder = '../log', gif=False, duration=0.33):
    import imageio
    lst, df_pids = traverse_panos_by_rid(rid, DB_panos, PANO_log, True)

    img_fns = []
    for i in tqdm(lst):
        pred, df_memo = lane_shape_predict(i, df_memo)
        img_fn = os.path.join(folder, pred['name'])
        img_fns.append(img_fn)
        draw_pred_lanes_on_img( pred, img_fn )

    if gif:
        images = []
        for filename in img_fns:
            images.append(imageio.imread(filename))
        imageio.mimsave(f'{folder}/{rid}.gif', images, duration=duration)
    
    return True


df_memo = pd.read_csv(config['data']['df_pred_memo'])
df_memo.loc[:, 'pred'] = df_memo.pred.apply( lambda x: eval(x) )
# df_memo.loc[:, 'name'] = df_memo.name.apply( lambda x: x.split('/')[-1] )
df_memo.to_csv(config['data']['df_pred_memo'], index=False)


img_fn = '9e9476-b801-afe8-b63b-221ce1_01_09005700121708261715532122S_297.jpg'
pred, df_memo = lane_shape_predict(img_fn, df_memo)
draw_pred_lanes_on_img( pred, pred['name'] )


rid = '7ea73e-734a-be1a-b9f4-2310d5'
rid = 'cf0303-a62d-1625-bed7-b74e6e'
# rid = 'b44ecb-2210-0f17-f060-1809ce'
rid = '1b8768-5c98-d5a0-218b-2fb5fb'




for i in res[:10]:
    _, df_memo = lane_shape_predict(i, df_memo)



#%%
# BUG 一条路仅有一个点的情况下，需要更新DIR数值； 终点也需要更新数值
DB_panos.query("RID=='168fb0-6c88-4d0b-59b0-20a829' ")


df_road_with_1_node = DB_panos[['RID', 'PID']].groupby("RID").count().rename(columns={"PID":'count'}).reset_index()
df_road_with_1_node.query("count==1")


from db.db_process import load_from_DB, extract_connectors_from_panos_respond

connecters = extract_connectors_from_panos_respond( DB_pano_base, DB_roads )

