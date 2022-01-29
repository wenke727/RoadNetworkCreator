#%%
import io
import os
import time
import random
import requests
import pandas as pd
import multiprocessing
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from utils.http_helper import get_proxy
from utils.minio_helper import MinioHelper
from setting import PANO_FOLFER, LOG_FOLDER
from utils.geo_plot_helper import map_visualize
from utils.parallel_helper import parallel_process

minio_helper = MinioHelper()

from loguru import logger
logger.remove()
logger.add(
    os.path.join(LOG_FOLDER, f"pano_img_{time.strftime('%Y-%m-%d', time.localtime())}.log"), 
    enqueue=True,  
    backtrace=True, 
    diagnose=True,
    level="INFO",
    mode='w'
)

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


""" new API 201010"""
def drop_pano_file(lst, folder=PANO_FOLFER):
    if isinstance(lst, gpd.GeoDataFrame):
        lst = lst.apply(lambda x: f"{x.PID}_{x.DIR}.jpg", axis=1).values.tolist()
    
    for i in tqdm(lst, desc="Remove existing files"):
        fn = os.path.join(folder, i)
        if not os.path.exists(fn):
            continue
        os.remove(fn)
    
    pass


def fetch_pano_img(pid, heading, rewrite=False, path=None, proxies=True, logger=logger):
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
    
    if not rewrite and minio_helper.file_exist(path): 
        return path

    def __save(file):
        f = open(path, 'wb')
        f.write(file.content)
        f.flush()
        f.close()
    
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=55&quality=100&panoid={pid}&heading={heading}&pitch=-10&width=1024&height=576"
    try:
        file = requests.get(url, timeout=60, proxies={'http': get_proxy()} if proxies else None  )
            
        if logger is not None: 
            logger.info( f"{pid}: {url}")
        
        if False: 
            __save(file)
        
        buf = io.BytesIO(file.content)
        ret_dict = minio_helper.upload_file(file=buf, object_name=path)
        path = ret_dict['public_url']

        if not proxies: 
            time.sleep( random.triangular(0.1, 3, 10) )
        
        return path

    except:
        if logger is not None: 
            logger.error(f'crawled url failed: {url} ')
        time.sleep( random.uniform(30, 180) )

    return None


def fetch_pano_img_parallel(lst, heading_att='DIR', rewrite=True, n_jobs=32, with_bar=True, bar_describe="Crawl panos"):
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
        result.append( pool.apply_async(fetch_pano_img, (item['pid'], item['heading'], rewrite, ), callback=update) )
    pool.close()
    pool.join() 

    result = [i.get() for i in result]

    return result


""" new API 210114"""
def pano_img_api(pid, heading, fovy=56, pitch=-10, quality=100):
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&quality={quality}&panoid={pid}&pitch={pitch}&fovy={fovy}&width=1024&height=576&heading={heading}"
    
    return url
    

def _request_helper(url, proxies=True, timeout=10):
    i = 0
    while i < 3:   
        try:
            proxy = {'http': get_proxy()} if proxies else None
            respond = requests.get(url, timeout=timeout, proxies=proxy)
            
            if respond.status_code == 200:
                logger.info(f'{url}')
                return respond

            logger.warning( f"{url}, status_code: {respond.status_code}")
            continue

        except requests.exceptions.RequestException as e:
            logger.warning(f"{url}, RequestException: ", e)

        i += 1
        if not proxies:
            time.sleep(random.randint(1,10))
    
    logger.error(f"{url}, try 3 times but failed")
    
    return None


def get_pano_img(pid, heading, overwrite=True, plot=False):
    def _plot():
        plt.imshow(Image.open(path))
        plt.axis('off')
        
    url = pano_img_api(pid, heading)

    # folder = os.path.join(PANO_FOLFER, {heading})
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    fn = f"{pid}_{heading}.jpg"
    path = os.path.join(PANO_FOLFER, f"{pid}_{heading}.jpg")

    if  not overwrite and minio_helper.file_exist(fn):
        if plot: _plot()
        return path

    res = _request_helper(url)
    if res is not None:
        buf = io.BytesIO(res.content)
        ret_dict = minio_helper.upload_file(file=buf, object_name=fn)
        assert 'public_url' in ret_dict, f"{pid}, {heading}, {ret_dict}"

        if plot: 
            _plot()

        return ret_dict['public_url'] if 'public_url' in ret_dict else None

    return None


def get_pano_img_parallel(lst, overwrite):
    # lst = [("01005700001404191412515725W", 190), ("01005700001404191412515725W", 191)]
    params = [(i, d, overwrite) for i, d in lst]
    res = parallel_process(get_pano_img, params, True, desc="Parallel fetching imgs")
    
    return res


#%%

if __name__ == "__main__":

    """" check for new API """
    pid = '01005700001312021447154435T'
    heading = 180
    
    param = {'pid': '01005700001312021447154435T', 'heading': 180}
    res = fetch_pano_img_parallel([param, param])
    print(res)


    #%%
    # ! 更新记录，删除已有文件，预测图片
    panos = gpd.read_file("./nanguang_traj0.geojson")

    lst = panos[['PID', "MoveDir"]].values 
    res = get_pano_img_parallel(lst, overwrite=True)
    
    
    # delete exist imgs
    # drop_pano_file(panos.query("DIR != MoveDir"))

    # download the new imgs
    # res = fetch_pano_img_parallel(panos, heading_att = 'MoveDir')

