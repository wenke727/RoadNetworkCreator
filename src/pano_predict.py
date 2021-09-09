#%%
"""
# TODO step 5: predict pano imgs
    拼接出图
    实现预测的功能，并绘制结果
    基本的过滤 和 数据处理
"""
import os
import math
from matplotlib.pyplot import legend
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import Point
import warnings

from utils.df_helper import load_df_memo, query_df
from utils.img_process import plt_2_Image, cv2_2_Image, combine_imgs
from utils.geo_plot_helper import map_visualize
from model.lstr import draw_pred_lanes_on_img, lstr_pred, lstr_pred_by_pid

from setting import PANO_FOLFER, PRED_MEMO

warnings.filterwarnings('ignore')


#%%
# 获取基础的数据
def get_panos_imgs_by_bbox(bbox=[113.92348,22.57034, 113.94372,22.5855], vis=True, with_folder=False):
    """给定一个区域，获取所有的panos

    Args:
        bbox (list, optional): [description]. Defaults to [113.92348,22.57034, 113.94372,22.5855].
        vis (bool, optional): [description]. Defaults to True.
        with_folder (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    pass


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


# 预测
def lane_shape_predict(img_fn, df_memo):
    """针对img_fn的照片进行车道线形预测，并将结果缓存，便于再次访问

    Args:
        img_fn ([type]): [description]
        df_memo ([df]): prediction memoization.

    Returns:
        [type]: [description]
    """
    img_name = img_fn.split('/')[-1]

    # TODO 直接改成索引，降低query的影响
    if not df_memo.query( f" name == '{img_name}' " ).shape[0]:
        print(img_name)
        info = lstr_pred( img_name )
        info['DIR'] = info['heading']
        df_memo = df_memo.append( info, ignore_index=True )

        return info

    return df_memo.query( f" name == '{img_name}' " ).to_dict('records')[0]


def update_unpredict_panos(pano_lst, DB_panos, df_memo):
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
    queue = unpredict_lst.apply(lambda x: f"{x.PID}_{x.DIR}.jpg", axis=1)
    
    for i in tqdm( queue.values, 'update_unpredict_panos'):
        _, df_memo = lane_shape_predict(i, df_memo)
    pano_lst = pano_lst.merge(df_memo, how='left', on='name')
    
    
    return pano_lst, df_memo


def pred_trajectory(gdf, df_memo, resize_factor=.5, aerial_view=True, combine_view=False, with_lanes=True):
    res = {}
    res['gdf'] = gdf.merge(df_memo[['PID', 'DIR','lane_num','name', 'pred']], on=['PID', 'DIR'])
    fn_lst = res['gdf'].apply(lambda x: os.path.join(PANO_FOLFER, f"{x['name']}"), axis=1).values.tolist()


    if combine_view:
        # TODO: add position; add debug info
        if with_lanes:
            img_lst = [draw_pred_lanes_on_img(lane_shape_predict(fn, df_memo), fn) for fn in fn_lst if os.path.exists(fn)]
        else:
            img_lst = [Image.open(fn) for fn in fn_lst if os.path.exists(fn)]

        comb_img = combine_imgs(img_lst)

        x, y = comb_img.size
        comb_img = comb_img.resize((int(resize_factor*x), int(resize_factor*y)))

        res['combine_view'] = comb_img

    if aerial_view:
        fig, ax = map_visualize(res['gdf'], color='gray')
        gdf_plot = res['gdf'].query('lane_num >= 2')
        gdf_plot.loc[:, 'lane_num'] = gdf_plot.loc[:, 'lane_num'].astype('str')
        gdf_plot.plot(column='lane_num', legend=True, categorical=False, ax=ax)
        res['aerial_view'] = fig
        plt.close()

    return res


#%%

if __name__ == '__main__':
    df_memo = load_df_memo(PRED_MEMO)
    gdf = gpd.read_file("../cache/panos_for_test.geojson")
    res = pred_trajectory(gdf)
    # res['combine_view']

    res['aerial_view']

# %%
