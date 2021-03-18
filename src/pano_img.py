# from main import *
from tqdm import tqdm
import random
import time
import urllib 
import os

# from PIL import Image
import seaborn as sns
from db.db_process import load_from_DB
from utils.log_helper import LogHelper
from utils.utils import load_config
import logbook
from utils.geo_plot_helper import map_visualize

STEPS = 4

config    = load_config()
pano_dir  = config['data']['pano_dir']
input_dir = config['data']['input_dir']
PANO_log = LogHelper(log_dir=config['data']['log_dir'], log_name='pano_img.log').make_logger(level=logbook.INFO)




def get_staticimage(pid, heading, path, log_helper=None, sleep=True):
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
    
    # id = "09005700121902131650290579U"; heading = 87
    if os.path.exists(path): return path

    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={pid}&heading={heading}&pitch=0&width=1024&height=576"
    request = urllib.request.Request(url=url, method='GET')
    try:
        file = urllib.request.urlopen(request, timeout = 60)
        if sleep: time.sleep( random.triangular(0.1, 3, 10) )
        if log_helper is not None: log_helper.info( f"{pid}: {url}")
        
        f = open(path, 'wb')
        f.write(file.read())
        f.flush()
        f.close()
    except:
        if log_helper is not None: 
            log_helper.error(f'crawled url failed: {url} ')
        time.sleep( random.uniform(30, 180) )

    # except urllib.error as e:
    #     # FIXME http.client.IncompleteRead: IncompleteRead(114291 bytes read, 502762 more expected), 
    #     if log_helper is not None: 
    #         if hasattr( e, 'code' ):
    #             log_helper.error(f'crawled url failed: {url}, {e.code} ')
    #         if hasattr( e, 'reason' ):
    #             log_helper.error(f'crawled url failed: {url}, for {e.reason}')
    # except Exception as e:
    #     log_helper.error(f'crawled url failed: {url}, {str(e)}')

    # except http_client.IncompleteRead as e:
    #     if log_helper is not None: 
    #         if hasattr( e, 'code' ):
    #             log_helper.error(f'crawled url failed: {url}, IncompleteRead ')
    return path


def get_pano_ids_by_rid(rid, DB_panos, vis=False):
    tmp = DB_panos.query( f" RID == '{rid}' " )
    if vis: map_visualize(tmp)
    return tmp


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



def traverse_panos(df_panos):
    # db_process.py
    import time, random
    from tqdm import tqdm

    # 遍历照片
    RIDs = df_panos.RID.unique()
    for rid in  RIDs[:]:
        df = df_panos.query( f"RID == '{rid}' " )

        for index, item in df.iterrows():
            if not (item.Order == 0 or item.Order == df.shape[0]-2):
                continue
            res = get_staticimage( item.PID, item.DIR )
            if res is not None:
                time.sleep(random.uniform(2, 5))


def query_static_imgs_by_road(name = '光侨路'):
    # 根据道路获取其街景
    # TODO 推送到minio服务器

    rids = DB_roads.query(f"Name == '{name}' ").RID.unique().tolist()
    df = DB_panos.query( f"RID in {rids}" )

    for index, item in tqdm( df[130:].iterrows()):
        if item.DIR == 0:
            continue
        
        if get_staticimage(item.PID, heading=item.DIR):
            time.sleep(random.triangular(0.5, 1, 10))

    return True



if __name__ == "__main__":
    # rid = "7b3a55-bab4-becf-aea3-a9344d"
    # traverse_panos_by_rid(rid)
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
    
    rids = DB_roads.RID.unique()

    for rid in tqdm( rids[::-1] ):
        traverse_panos_by_rid(rid=rid, DB_panos=DB_panos, log=PANO_log)
    pass
