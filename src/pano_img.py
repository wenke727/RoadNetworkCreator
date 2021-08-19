#%%
import io
import os
import requests
import time
import random
import multiprocessing
import logbook
from utils.http_helper import get_proxy
from utils.minio_helper import MinioHelper
from utils.log_helper import LogHelper

minio_helper = MinioHelper()
pano_API_log = LogHelper(log_dir="../log", log_name='panos_base.log').make_logger(level=logbook.INFO)


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
    """Old API for traverse panos

    Args:
        df_panos ([type]): [description]
    """
    import time, random
    from tqdm import tqdm

    # 遍历照片
    RIDs = df_panos.RID.unique()
    for rid in  RIDs[:]:
        df = df_panos.query( f"RID == '{rid}' " )

        for index, item in df.iterrows():
            if not (item.Order == 0 or item.Order == df.shape[0]-2):
                continue
            res = get_staticimage( item.PID, item.DIR, pano_dir )
            if res is not None:
                time.sleep(random.uniform(2, 5))


def query_static_imgs_by_road(name, pano_dir):
    # 根据道路获取其街景
    rids = DB_roads.query(f"Name == '{name}' ").RID.unique().tolist()
    df = DB_panos.query( f"RID in {rids}" )

    for index, item in tqdm( df[130:].iterrows()):
        if item.DIR == 0:
            continue
        
        if get_staticimage(item.PID, heading=item.DIR):
            time.sleep(random.triangular(0.5, 1, 10))

    return True


""" new API """

def get_staticimage(pid, heading, path=None, log_helper=pano_API_log, sleep=True, proxy='pool'):
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
            
        if log_helper is not None: 
            log_helper.info( f"{pid}: {url}")
        
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
        if log_helper is not None: 
            log_helper.error(f'crawled url failed: {url} ')
        time.sleep( random.uniform(30, 180) )

    return None


def _multi_helper(param):
    res = get_staticimage(**param)

    return res


def get_staticimage_batch(lst, n_jobs=30, with_bar=True, bar_describe="Crawl panos"):
    if lst is None:
        return None
    
    from tqdm import tqdm
    pbar = tqdm(total=len(lst))
    if bar_describe is not None:
        pbar.set_description(bar_describe)
    update = lambda *args: pbar.update() if with_bar else None
    
    pool = multiprocessing.Pool(n_jobs)
    result = pool.map_async(_multi_helper, lst, callback=update).get()
    pool.close()
    pool.join() 

    return result


#%%

if __name__ == "__main__":
    # rid = "7b3a55-bab4-becf-aea3-a9344d"
    # traverse_panos_by_rid(rid)
    # DB_roads = load_DB_roads()
    # rids = DB_roads.RID.unique()

    # for rid in tqdm( rids[::-1] ):
    #     traverse_panos_by_rid(rid=rid, DB_panos=DB_panos, log=PANO_log)
    # pass

    """" check for new API """
    pid = '01005700001312021447154435T'
    heading = 180
    
    param = {'pid': '01005700001312021447154435T', 'heading': 180}
    _multi_helper(param)
    res = get_staticimage_batch([param, param])
    
    print(res)

