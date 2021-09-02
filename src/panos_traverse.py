#%%
import os
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt

from db.features_API import get_features
from db.db_process import load_postgis, gdf_to_postgis
from utils.geo_plot_helper import map_visualize
from pano_base import pano_dict_to_gdf, update_move_dir
from pano_img import get_staticimage, get_staticimage_batch, get_pano_ids_by_rid

from setting import PANO_FOLFER

#%%


""" origin """

def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
    # TODO
    """Obtain the panos in road[rid] by distributed crawlers.

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
        if heading == 0 and id != 0:
            heading = pre_heading
        
        if not all:
            if length > 3 and order % 3 != 1:
                continue

        # Fetch data asynchronously
        fn = f"{PANO_FOLFER}/{pid}_{heading}.jpg"
        params = {'pid': pid, 'heading': heading}
        if os.path.exists(fn):
            continue

        result = get_staticimage(**params)
        res.append(result)
        pre_heading = heading
        
    return res, df_pids


def count_panos_num_by_area():
    """Count panos number in several area.

    Returns:
        [type]: [description]
    """
    areas = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs_with_Dapeng.geojson')
    res = {}
    for index, district in tqdm(areas.iterrows()):
        panos = get_features('point', geom = district.geometry)
        res[district.name_cn] = panos.shape[0]
        
    # df = gpd.sjoin(left_df=area, right_df=DB_panos, op='contains').groupby('name')[['DIR']].count()

    return res


def crawl_panos_in_district_area(district='南山区', key='name', fn='/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson'):
    area = gpd.read_file(fn)
    area.query( f"{key} =='{district}'", inplace=True )
    if area.shape[0] == 0:
        return None
    
    area = area.iloc[0].geometry
    features = get_features('line', geom = area)
    panos = get_features('point', geom = area)

    res    = []
    count = 0
    for rid in tqdm(features.RID.unique(), district):
        info, _ = traverse_panos_by_rid(rid, panos, log=None, all=True)
        res += info
        count += 1
        # if count > 500: break
    print(len(res))
    
    return res


def crawl_panos_in_city_level(fn='/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs_with_Dapeng.geojson'):
    areas = gpd.read_file(fn)
    for index, district in areas.iterrows():
        crawl_panos_in_district_area(district.name_cn, 'name_cn', fn)
    
    return True


""" new API """

def batch_name_change():
    """Change the format of pano file name from `rid_order_pid_heading` to `pid_heading`
    """
    for fn in tqdm(os.listdir(PANO_FOLFER)):
        if 'jpg' not in fn:
            continue
        
        old_name = os.path.join(PANO_FOLFER, fn)
        new_name = os.path.join(PANO_FOLFER, "_".join(fn.split("_")[2:]))
        
        os.rename(old_name, new_name)
    
    return


def load_exist_pano_imgs(attrs=['pid', 'dir']):
    """load exit pano file in `PANO_FOLFER`

    Args:
        attrs (list, optional): [description]. Defaults to ['pid', 'dir'].

    Returns:
        [lst]: The list of pano imgs.
    """
    fn_lst = os.listdir(PANO_FOLFER)
    fn_lst =  pd.DataFrame([ [i] + i.split("_") for i in  fn_lst if 'jpg' in i ], columns=['fn']+attrs)
    fn_lst.loc[:, 'dir'] = fn_lst.dir.apply(lambda x: int(x[:-4]))

    return fn_lst


def check_unvisited_pano_img(pano_dict:dict):
    """Check for unvisited pano img data

    Args:
        pano_dict (dict): The dict of panos

    Returns:
        [list]: The unvisited pano records.
    """

    pano_imgs_exist = load_exist_pano_imgs()
    pano_ids_exist = set( pano_imgs_exist.pid.unique() )

    new_pano = { key: pano_dict[key] for key in pano_dict if key not in pano_ids_exist }
    
    if len(new_pano) == 0:
        return None

    new_pano = pano_dict_to_gdf(new_pano)
    new_pano.loc[:, 'pid'] = new_pano.index
    new_pano.loc[:, 'heading'] = new_pano.MoveDir.astype(int)
    new_pano = new_pano[['pid', 'heading']].to_dict('records')

    return new_pano


def delete_history_error_dir_panos(download_new_pano=True, update=False):
    """
    Deleted photos with incorrect azimuth due to historical reasons
    """

    df_panos = load_postgis('panos')
    df_panos.set_index('PID', inplace=True)

    df_key_panos = load_postgis('pano_base')
    df_key_panos.set_index('ID', inplace=True)

    df_panos.loc[:, 'DIR_bak'] = df_panos.DIR
    df_panos = update_move_dir(df_panos, df_key_panos)


    rm_lst = df_panos.query("DIR != -1 and DIR != DIR_bak ").reset_index()[['PID', 'DIR_bak']].values.tolist()

    remove_count = 0
    for pid, heading in tqdm(rm_lst):
        try:
            os.remove(os.path.join(PANO_FOLFER, f"{pid}_{heading}.jpg"))
            remove_count += 1
        except:
            continue
    
    print(f"Delete {remove_count} pano imgs")

    if download_new_pano:
        pano_imgs_exist = load_exist_pano_imgs()

        df_panos = df_panos.drop(columns=['DIR_bak', 'url']).reset_index().drop_duplicates(['PID', 'DIR'])
        pano_lst = df_panos[['PID', 'DIR']].rename(columns={'PID':'pid', 'DIR':'heading'})
        pano_lst = pano_lst.merge(pano_imgs_exist, left_on=['pid', 'heading'], right_on=['pid', 'dir'], how='left')
        pano_lst = pano_lst[pano_lst.fn.isna()][['pid', 'heading']].to_dict('records')

        get_staticimage_batch(pano_lst)

    if update:
        df_panos = df_panos.set_index('PID')
        df_panos.loc[:, 'PID'] = df_panos.index
        df_panos.loc[:, 'url']  = df_panos.apply(lambda x: f"http://192.168.135.15:9000/panos/{x.PID}_{x.DIR}.jpg", axis=1)
        attrs_order= ['PID', 'DIR', 'RID', 'Order', 'Type', 'X', 'Y', 'geometry', 'url', 'lane_num']

        gdf_to_postgis(df_panos[attrs_order], 'panos')

    return 


#%%

if __name__ == '__main__':
    # crawl_panos_in_district_area('盐田区')
    # count_panos_num_by_area()  
    {'福田区': 82795,
    '罗湖区': 46297,
    '南山区': 130939,
    '盐田区': 23611,
    '大鹏新区': 20622,
    '龙华区': 98013,
    '坪山区': 43334,
    '龙岗区': 142577,
    '宝安区': 163176,
    '光明新区': 44404}


    """crawl at city level"""
    crawl_panos_in_city_level()


    """get unvisited pano"""
    from utils.pickle_helper import PickleSaver
    saver = PickleSaver()
    pano_dict = saver.read('../cache/pano_dict_szu.pkl')
    new_pano_lst = check_unvisited_pano_img(pano_dict)

    get_staticimage_batch(new_pano_lst)
    
    # check
    param = {'pid': '01005700001312021447154435T', 'heading': 180}
    get_staticimage_batch([param, param, param])
    
    """ delete panos with error direction, and download the correct one """
    delete_history_error_dir_panos()
