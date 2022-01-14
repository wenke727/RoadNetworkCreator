#%%
import io
import os
import time
import random
import logbook
import requests
import pandas as pd
import multiprocessing
import geopandas as gpd
from tqdm import tqdm
from utils.log_helper import LogHelper
from utils.http_helper import get_proxy
from utils.minio_helper import MinioHelper
from setting import PANO_FOLFER
from utils.geo_plot_helper import map_visualize

minio_helper = MinioHelper()


#%%

""" origin """

def get_pano_ids_by_rid(rid, DB_panos, vis=False):
    """Get panos ids in a road segement by its name.

    Args:
        rid (str): road id
        DB_panos ([type]): [description]
        vis (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    tmp = DB_panos.query( f" RID == '{rid}' " )
    if vis: map_visualize(tmp)
    
    return tmp


def traverse_panos_by_rid(rid, DB_panos, log=None, all=False):
    """obtain the panos in road[rid], sampling is carried out for the relatively long raod link.

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
                continue

        fn = f"{PANO_FOLFER}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        res.append(fetch_pano_img(pid=pid, heading=heading, path=fn, logger=log))
        pre_heading = heading
        
    return res, df_pids


""" new API """
def drop_pano_file(lst, folder=PANO_FOLFER):
    if isinstance(lst, gpd.GeoDataFrame):
        lst = lst.apply(lambda x: f"{x.PID}_{x.DIR}.jpg", axis=1).values.tolist()
    
    for i in tqdm(lst):
        fn = os.path.join(folder, i)
        if not os.path.exists(fn):
            continue
        os.remove(fn)
    
    pass


def fetch_pano_img(pid, heading, path=None, sleep=True, proxy='pool', logger=None,):
    """get static image from Baidu View with `pano id`

    Args:
        pid ([type]): [description]
        heading ([type]): [description]
        path ([type]): [description]
        sleep (bool, optional): [description]. Defaults to True.
        log (LogHelper, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if path is None:
        path = f"{pid}_{heading}.jpg"
    else:
        path = path.split('/')[-1]
    
    if minio_helper.file_exist(path): 
        return path

    def __save(file):
        f = open(path, 'wb')
        f.write(file.content)
        f.flush()
        f.close()
    
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={pid}&heading={heading}&pitch=0&width=1024&height=576"
    try:
        if proxy is None:
            file = requests.get( url, timeout=60 )
        else:
            file = requests.get( url, timeout=60, proxies={'http': get_proxy()}  )
            
        if logger is not None: 
            logger.info( f"{pid}: {url}")
        
        if False: 
            __save(file)
        # ret_dict = minio_helper.upload_file(file_path=path, object_name=path)
        # if os.path.exists(path):  os.remove(path) 
        
        buf = io.BytesIO(file.content)
        ret_dict = minio_helper.upload_file(file=buf, object_name=path)
        path = ret_dict['public_url']

        if sleep: 
            time.sleep( random.triangular(0.1, 3, 10) )
        
        return path

    except:
        if logger is not None: 
            logger.error(f'crawled url failed: {url} ')
        time.sleep( random.uniform(30, 180) )

    return None


def fetch_pano_img_parallel(lst, heading_att='DIR', n_jobs=32, with_bar=True, bar_describe="Crawl panos"):
    if lst is None:
        return None
    
    if isinstance(lst, gpd.GeoDataFrame) or isinstance(lst, pd.DataFrame):
        lst = lst.reset_index().\
                  rename(columns={"PID": 'pid', heading_att: 'heading'})[['pid', 'heading']].\
                  to_dict(orient='records')
    
    pbar = tqdm(total=len(lst), desc='Parallel fetching staticimage: ')
    if bar_describe is not None:
        pbar.set_description(bar_describe)
    update = lambda *args: pbar.update() if with_bar else None
    
    pool = multiprocessing.Pool(n_jobs)
    result = []
    for item in lst:
        result.append( pool.apply_async(fetch_pano_img, (item['pid'], item['heading'],), callback=update) )
    pool.close()
    pool.join() 

    result = [i.get() for i in result]

    return result


#%%

if __name__ == "__main__":
    pano_API_log = LogHelper(log_dir="../log", log_name='panos_base.log').make_logger(level=logbook.INFO)

    """" check for new API """
    pid = '01005700001312021447154435T'
    heading = 180
    
    param = {'pid': '01005700001312021447154435T', 'heading': 180}
    res = fetch_pano_img_parallel([param, param])
    print(res)


    #%%
    # ! 更新记录，删除已有文件，预测图片
    panos = gpd.read_file("./jintianroad_north.geojson")

    # delete exist imgs
    drop_pano_file(panos.query("DIR != MoveDir"))

    # download the new imgs
    res = fetch_pano_img_parallel(panos, heading_att = 'MoveDir')

