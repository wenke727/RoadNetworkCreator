from matplotlib.pyplot import flag
from shapely import geometry
from tqdm.cli import main
from main import *
from mapAPI import get_staticimage
import yaml
from tqdm import tqdm
import random
import time
import http

# from PIL import Image
import seaborn as sns
from utils.log_helper import LogHelper
import logbook

STEPS = 4
with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
    config = yaml.load( f )
pano_dir  = config['data']['pano_dir']
input_dir = config['data']['input_dir']
PANO_log = LogHelper(log_dir=config['data']['log_dir'], log_name='panos.log').make_logger(level=logbook.INFO)


def get_staticimage(pid, heading, path, log_helper=None, sleep=True):
    """[summary]

    Args:
        pid ([type]): [description]
        heading ([type]): [description]
        path ([type]): [description]
        sleep (bool, optional): [description]. Defaults to True.
        log (LogHelper, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # origin: /home/pcl/traffic/RoadNetworkCreator_by_View/src/mapAPI.py
    # id = "09005700121902131650290579U"; heading = 87
    if os.path.exists(path): return path

    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={pid}&heading={heading}&pitch=0&width=1024&height=576"
    request = urllib.request.Request(url=url, method='GET')
    try:
        file = urllib.request.urlopen(request, timeout = 60)
        if sleep: time.sleep( random.triangular(0.1, 3, 10) )
        if log_helper is not None: log_helper.info(url)
        
        f = open(path, 'wb')
        f.write(file.read())
        f.flush()
        f.close()
    except urllib.error as e:
        # FIXME http.client.IncompleteRead: IncompleteRead(114291 bytes read, 502762 more expected), 
        if log_helper is not None: 
            if hasattr( e, 'code' ):
                log_helper.error(f'crawled url failed: {url}, {e.code} ')
            if hasattr( e, 'reason' ):
                log_helper.error(f'crawled url failed: {url}, for {e.reason}')
    except Exception as e:
        log_helper.error(f'crawled url failed: {url}, {str(e)}')

    # except http_client.IncompleteRead as e:
    #     if log_helper is not None: 
    #         if hasattr( e, 'code' ):
    #             log_helper.error(f'crawled url failed: {url}, IncompleteRead ')
    return path

def get_pano_id_by_rid(rid, vis=False):
    tmp = DB_panos.query( f" RID == '{rid}' " )
    if vis: map_visualize(tmp)
    return tmp

def traverse_panos_by_rid(rid, log=None):
    """obtain the panos in road[rid] 

    Args:
        rid (str): the id of road segements

    Returns:
        [type]: [description]
    """
    df_pids = get_pano_id_by_rid(rid)

    # FIXME the strategy to crawl the panos data
    pano_lst = df_pids.query( f"Order % @STEPS == 3 and Order != {df_pids.Order.max()}" )[['Order','PID', 'DIR']].values
    for id, (order, pid, heading) in enumerate(pano_lst):
        fn = f"{pano_dir}/{rid}_{order:02d}_{pid}_{heading}.jpg"
        # print(fn)
        get_staticimage( pid=pid, heading=heading, path=fn, log_helper=log )

    return 

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

if __name__ == "__main__":
    # rid = "7b3a55-bab4-becf-aea3-a9344d"
    # traverse_panos_by_rid(rid)
    rids = DB_roads.RID.unique()

    for rid in tqdm( rids[::-1] ):
        traverse_panos_by_rid(rid=rid, log=PANO_log)
    pass
