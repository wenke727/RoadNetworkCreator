import os, sys
import math
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from road_network import OSM_road_network

from db.features import get_features
from db.db_process import load_from_DB
from road_matching import traverse_panos_by_rid
from pano_img import get_pano_ids_by_rid, get_staticimage, pano_dir
from utils.geo_plot_helper import map_visualize
from pano_img import PANO_log

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
    folder = './images'
    dst    = "~/Data/TuSimple/LSTR/lxd"
    res    = []
    
    # features = get_features('line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])
    bbox=[113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
    # bbox=[113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
    # bbox = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
    features = get_features('line', bbox=bbox)
    map_visualize(features)

    for rid in tqdm(features.RID.unique()):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log, all=True)
        res += info
    len(res)
    
    
    rids = ['ee0530-118d-1788-6960-40a917', '3e9933-e732-1db9-61dc-ee8b54',
       '69a92f-4240-0901-4dfb-42b25a', '9ac2be-1409-219f-8eae-045791',
       '7a88c4-e2de-1f0a-ce71-210b8f', '9aa056-13a8-7161-f27f-29bdc0',
       '00046f-a910-b5a7-a510-882479', '9b6a92-19d2-c73b-b682-a12e82',
       '195c6f-aaf7-fb2f-ddbb-df8f5a', 'd9eb18-4ee1-d36c-f6ef-74f80d',
       '31b7c3-40c5-f0d3-5716-a19099', 'dfbb1c-5d1b-dd32-f0f6-e73b50',
       '697020-7aff-d171-b4e5-0738e2', '091bf8-720c-19ca-50a2-d230fd',
       'bb985a-e5ce-4c49-17b0-3c8152', '1e701a-5974-1c43-beeb-1e97ba']
    

    for rid in tqdm(rids):
        info, _ = traverse_panos_by_rid(rid, DB_panos, log=PANO_log, all=True)
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




#! 获取某一个区域所有的panos






#%% 
fn = "../lxd_predict.csv"

pd.read_csv(fn)

# %%
df.lane_num.value_counts()
# %%




rid = 'a17f95-7a3f-2fe5-a797-ec84d5'
rid = '2a8f95-3565-ebb2-7fcd-04efd5'
rid = '7ea73e-734a-be1a-b9f4-2310d5'
res, df_pids = traverse_panos_by_rid(rid, DB_panos, PANO_log, True)
print(res)

df_pids


#%%
# 预测模块


from model.lstr import draw_pred_lanes_on_img, lstr_pred

def lane_shape_predict(img_fn, df_memo):
    
    img_name = img_fn.split('/')[-1]
    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        print(f"query data: {img_name}")
        info = lstr_pred( img_name )
        info['respond']['pid'] = img_name.split('_')[2]
        df_memo = df_memo.append( info['respond'], ignore_index=True )

        return info['respond'], df_memo

    return df_memo.query( f" name == '{img_name}' " ).to_dict('records')[0], df_memo


df_memo = pd.read_csv('../lxd_predict.csv')

img_fn = '/home/pcl/Data/minio_server/panos/18d667-0e65-ebb2-7fcd-04efee_00_09005700121709091110018169Y_0.jpg'
info = lstr_pred( img_fn )
draw_pred_lanes_on_img( info['respond'], 'test1.jpg' )

for i in tqdm(lst):
    _, df_memo = lane_shape_predict( i, df_memo )


