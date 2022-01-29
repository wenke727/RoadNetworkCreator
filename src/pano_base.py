#%%
import io
import os
import sys
import math
import json
import time
import random
import shapely
import requests
import pinyin
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import geopandas as gpd
from loguru import logger
from copy import deepcopy
from shutil import copyfile
from collections import deque
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box


from baidu_map import get_shp_by_name_with_search_API
from panos_topo import combine_rids, Pano_UnionFind
from pano_img import get_pano_img_parallel

from utils.pickle_helper import PickleSaver
from utils.minio_helper import MinioHelper
from db.db_process import gdf_to_geojson, load_postgis, gdf_to_postgis
from utils.http_helper import get_proxy
from utils.coord.coord_transfer import bd_mc_to_wgs, wgs_to_bd_mc
from utils.geo_plot_helper import map_visualize
from utils.parallel_helper import parallel_process
from utils.geo_helper import geom_buffer


from utils.azimuth_helper import azimuth_diff
from setting import FT_BBOX, PANO_FOLFER, SZ_BBOX, GBA_BBOX, PCL_BBOX, LXD_BBOX, CACHE_FOLDER, SZU_BBOX, LOG_FOLDER, link_type_no_dict

import warnings
warnings.filterwarnings('ignore')

saver = PickleSaver()
logger.remove()
logger.add(
    os.path.join(LOG_FOLDER, f"pano_base_{time.strftime('%Y-%m-%d', time.localtime())}.log"), 
    enqueue=True,  
    backtrace=True, 
    diagnose=True,
    level="INFO",
    mode='w'
)


""" Pano API """
def query_pano(lon=None, lat=None, pid=None, details=False, memo=None, proxies=True):

    """[summary]

    Args:
        lon ([type], optional): [description]. Defaults to None.
        lat ([type], optional): [description]. Defaults to None.
        pid ([type], optional): [description]. Defaults to None.
        proxies (bool, optional): [description]. Defaults to True.
        memo ([dict], optional): [description]. Defaults to None.
        details (bool, optional): Query the pid info or not. Defaults to True.

    Return:
        {'pid': pid, 'info': item}
    """
    assert not(lon is None and lat is None and pid is None), "Check input"
    def _parse_pano_respond(res):
        content = res['content']
        assert len(content) == 1, logger.error(f"check result: {content}")
        
        item = content[0]
        if "X" not in item:
            assert "X" in item, print(f"check: {item}")
            logger.error(f'X not in respond, check: {item}')
            return None, None
        
        item['geometry'] = Point(bd_mc_to_wgs(item['X'], item['Y']))
        pid = item['ID']
        del item["ID"]
    
        return pid, item

    def _request_helper(url):
        i = 0
        while i < 3:   
            try:
                proxy = {'http': get_proxy()} if proxies else None
                respond = requests.get(url, timeout=5, proxies=proxy)
                
                if respond.status_code != 200:
                    logger.warning( f"{url}, error: {respond.status_code}")

                res = json.loads( respond.text )
                if 'result' in res and 'error' in res['result'] and res['result']['error'] == 404:
                    logger.warning(f"{url}, Respond: {res}")
                    return None
                
                if 'content' not in res:
                    logger.error(f"{url}, content not in the Respond: {res}")
                else:
                    logger.debug(f'fetch {url}, by {proxy}')
                return res

            except requests.exceptions.RequestException as e:
                logger.warning(f"{url}, RequestException: ", e)

            i += 1
            if not proxies:
                time.sleep(random.randint(1,10))
        
        logger.error(f"{url}, try 3 times but failed")
        
        return None
    
    url = None
    if lon is not None and lat is not None and pid is None:
        x, y = wgs_to_bd_mc(lon, lat)
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        res = _request_helper(url)
        
        if res is not None and 'content' in res and 'id' in res['content']:
            pid = res['content']['id']
            logger.info(f"Coords({lon}, {lat}) -> {pid}")
        else:
            logger.warning(f"Coords ({x}, {y}) has no matched pano.")
            return None
    
    if pid is not None:
        if memo is not None and pid in memo:
            return {'pid': pid, 'info': memo[pid]}
        if not details:
            return {'pid': pid}
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pid}"

    assert url is not None, "check the input"
    
    res = _request_helper(url)
    if res is None:
        return None
    pid, item = _parse_pano_respond(res)

    if memo is not None and pid is not None:
        memo[pid] = item
    
    return {'pid': pid, 'info': item}


def query_key_pano(lon=None, lat=None, pid=None, memo={}):
    key_panos = []
    respond = query_pano(lon, lat, pid=pid, proxies=True, memo=memo, details=True)
    
    if respond is None:
        return key_panos

    pid, record = respond['pid'], respond['info']
    for road in record['Roads']:
        if road['IsCurrent'] != 1:
            continue
        
        sorted(road['Panos'], key = lambda x: x['Order'])
        key_panos = panos = [road['Panos'][0]['PID']] + ([road['Panos'][-1]['PID']] if road['Panos'][0]['PID'] != road['Panos'][-1]['PID'] else [])
        for p in panos:
            if p in memo:
                continue
            
            if p != pid:  
                res = query_pano(pid=p, memo=memo, details=True)
                if res is None:
                    continue
            else:
                res = respond
            
            nxt, nxt_record = res['pid'], res['info']
            memo[nxt] = nxt_record

        break
    
    return key_panos


def fetch_helper(node, geofence, memo={}):
    """Coordinate with the parallel_process module to realize parallel fetching

    Args:
        node ([type]): [description]
        geofence ([type]): [description]
    
    Return: { 
        'pid': ...,
        'available': list, # outside the geofence
        'unavailable': list, # within the geofence
        'queue': list,
    }
    """
    res = {'pid': node, 'available':[], 'unavailable':[] ,'queue':[]}
    key_nodes = query_key_pano(pid=node, memo=memo)
    
    for pid in key_nodes:
        if pid not in memo:
            continue
        if geofence is not None and not memo[pid]['geometry'].within(geofence):
            res['unavailable'].append(pid)
            continue
        
        res['available'].append(pid)
        for link in memo[pid]['Links']:
            if link["PID"] in memo:
                continue
            
            res['queue'].append(link["PID"])
    
    return res, memo


def cp_imgs_to_dataset(road_name, df_trajs, traj_id, base_folder="/home/pcl/cv/LSTR/data/train", overwrite=False, sample=200):
    import random
    random.seed(42)
    road_name_py = pinyin.get(road_name, "_", format='strip')
    road_name_py = "".join([i[0].upper()+i[1:] for i in road_name_py.split("_")])
    folder = os.path.join(base_folder, road_name_py)

    df_traj = df_trajs.iloc[traj_id].pids_df
    
    # ax = map_visualize(df_traj)
    # plt.savefig(f"{base_folder}/{road_name_py}_{traj_id:02d}.jpg")
    
    res = self.parallel_fetch_img(df_traj, overwrite)
    size=len(res)
    gdf_to_geojson(df_traj, os.path.join(base_folder, f'{road_name_py}_{traj_id}'))

    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, img_fn in tqdm(enumerate(res)):
        if random.uniform(0, 1) > sample/size:
            continue
        if img_fn is None:
            continue
        
        img_fn = img_fn.split("/")[-1]
        # {traj_id:02d}_
        copyfile(os.path.join(PANO_FOLFER, img_fn), os.path.join(folder, f"{idx:04d}_{img_fn}") )
    
    return 


class OSMModule():
    def __init__(self, bbox=None, geofence=None):
        if bbox is not None:
            geofence = box(*bbox)
        assert geofence is not None, "Check Input"
        self.geofence = geofence
        
        self.osm_node_pg = "topo_osm_shenzhen_node"
        self.osm_edge_pg = "topo_osm_shenzhen_edge"
        
        self.link_type_no_dict = link_type_no_dict
        
        self.crs_wgs = 4326
        self.crs_prj = 900913
        
        pass


    def get_osm_node(self):
        self.osm_edge = load_postgis(self.osm_edge_pg, geom_wkt=self.geofence.to_wkt())
        helper = lambda x: self.link_type_no_dict[x] if x in self.link_type_no_dict else 99
        self.osm_edge.loc[:, "road_level"] = self.osm_edge.road_type.apply(helper)
        
        node_lst = np.unique(self.osm_edge.query('road_level < 6')[['s','e']].values.flatten()).tolist()
        self.osm_node = load_postgis(self.osm_node_pg, geom_wkt=self.geofence.to_wkt()).query('nid in @node_lst')

        return self.osm_node


    def osm_node_buffer(self, buffer_dis, att='buffer_zone'):
        if not hasattr(self, 'osm_node'):
            self.get_osm_node()
        if att in list(self.osm_node):
            return self.osm_node
        
        self.osm_node.loc[:, att] = self.osm_node.to_crs(epsg=self.crs_prj).buffer(buffer_dis).to_crs(epsg=self.crs_wgs)
        self.osm_node.index = self.osm_node.nid
        self.osm_node.set_geometry(att, inplace=True)

        return self.osm_node


class PanoCrawler(OSMModule):
    # TODO: 1) save and load; 
    def __init__(self, name, bbox=None, geofence:shapely.geometry=None, n_jobs=32, load_node=True):
        super().__init__(bbox=bbox, geofence=geofence)

        self.name = name
        self.osm_node = self.osm_node_buffer(10)
        self.minio_helper = MinioHelper()

        # http related
        self.n_jobs = n_jobs
        # proxypool
        self.proxy_func = get_proxy
        # db config
        self.pano_node_pg = 'pano_node'
        self.pano_link_pg = 'pano_link'
        
        # osm node
        self.buffer_dis  = 10
        self.coord_2_pid = {}
        self.dummy_pano  = None
        self.pano_dict   = self.load_pano_node() if load_node else {}
        self.osm_node    = self.osm_node_buffer(self.buffer_dis)
        self.osm_node.loc[:, 'visited'] = False
        
        assert hasattr(self, 'geofence') and self.geofence is not None, "Check Input: geofence"
        
        pass


    def load_pano_node(self):
        print('loading pano data:')
        gdf = load_postgis(self.pano_node_pg, geom_wkt=self.geofence.to_wkt())
        
        lst_atts = ['MoveDir']
        for att in lst_atts:
            gdf.loc[:, att] = gdf[att].astype(np.float)
        
        lst_atts = ['ImgLayer', 'Links', 'SwitchID', 'TimeLine', 'Roads',] 
        for att in tqdm(lst_atts, "Transfer"):
            gdf.loc[:, att] = gdf[att].apply(eval)
        
        gdf.index = gdf.ID
        
        return gdf.to_dict('index') 

    
    def crawl(self, overwrite=False):
        # TODO
        fn = os.path.join(CACHE_FOLDER, f"pano_dict_{self.name}.pkl")
        if os.path.exists(fn) and not overwrite:
            self.pano_dict = saver.read(fn)
        else:
            self.pano_dict = self._crawl_by_area()
            saver.save(self.pano_dict, fn)
        
        pass


    def _crawl_by_area(self, plot=True, queue_filter=True):
        start_time = time.time()
        queue = deque(self._get_unvisited_point())
        
        while queue:
            node = queue.popleft()
            ori_size = len(self.pano_dict)
            print(f"pano nums: {len(self.pano_dict)}, queue length: {len(queue)}, cur node: {node}")
            
            res = query_pano(*node)
            if res is None:
                continue

            pid = res['pid']
            if self.dummy_pano is None:
                self.dummy_pano = pid
            self._bfs(pid, memo=self.pano_dict, max_layer=100)
            self.coord_2_pid[node[:2]] = pid
            if len(self.pano_dict) == ori_size:
                continue
            
            self.gdf_pano_node = self.pano_dict_to_gdf(self.pano_dict)
            self.gdf_pano_link = self.extract_gdf_roads_from_key_pano(self.gdf_pano_node)
            lst = self._get_unvisited_point(self.gdf_pano_link)
            # FIXME
            lst = [i for i in lst if i not in self.coord_2_pid]

            if queue_filter:
                lst = self._crawl_queue_filter(lst=lst, visited=self.coord_2_pid)
                queue_filter = False
                
            queue = deque(lst)

        end_time = time.time()
        logger.info(f"time cost: {end_time-start_time:.1f} s")        

        if plot:
            map_visualize(self.gdf_pano_link)
        
        return

    # TODO
    def _crawl_by_point(self, ):
        # TODO
        pass


    def _bfs(self, pid, memo, max_layer=100):
        layer, queue = 0, [pid]
        size = ori_size = len(memo)

        logger.info(f"Fetching {pid} ({pid in self.pano_dict}): ")
        while True:
            res = self._parallel_fetch(queue, self.geofence, memo if layer==0 else {})
            if len(res) == 0:
                break
            query_info = [i[0] for i in res]
            new_panos  = [i[1] for i in res]

            for panos in new_panos:
                for k, v in panos.items():
                    memo[k] = v
            
            crawl_size = len(memo) - size
            total_size = len(memo) - ori_size
            size = len(memo)
            if crawl_size > 0:
                logger.info(f"{pid}, layer {layer:3d}, newly add {crawl_size:4d}, total {total_size:4d}")            

            df = pd.DataFrame(query_info).explode('queue')[['queue']].rename(columns={'queue': 'pid'}).drop_duplicates()
            df.loc[:, 'visited'] = df.pid.apply(lambda x: x in memo )
            queue = df.query('not visited and pid == pid').pid.values.tolist()

            if len(queue) == 0 or layer > max_layer:
                break
            layer += 1

        pass
        

    def _get_unvisited_point(self, gdf_roads:gpd.GeoDataFrame=None, plot=False):
        if gdf_roads is None or gdf_roads.shape[0]==0:
            return [(i[0], i[1]) for i in  self.osm_node[['x','y']].values.tolist()]

        visited = set(gpd.sjoin(left_df=self.osm_node.query('not visited'), right_df=gdf_roads, op='intersects')['nid'].unique())
        self.osm_node.loc[:, 'visited'] = self.osm_node.apply(lambda x: x['visited'] or x['nid'] in visited, axis=1)
        res = self.osm_node.query('not visited')
        
        if plot:
            ax = map_visualize(res, color='red')

        return [(i[0], i[1]) for i in  res[['x','y']].values.tolist()]


    def _parallel_fetch(self, queue, geofence, memo={}): 
        params_lst = [(pid, geofence, memo) for pid in queue]
        # print(f'memory size: {sys.getsizeof(params_lst)/1024:.2f}KB, ({(sys.getsizeof(params_lst)/sys.getsizeof(queue)-1)*100:.0f}% ⬆)')
        res = parallel_process(fetch_helper, params_lst, n_jobs=self.n_jobs)
        
        return res

    # TODO
    def _load_pg_pano(self):
        pass 


    def _crawl_queue_filter(self, lst, visited, pbar_switch=True, n_jobs=8, desc="Query coords: "):
        ori_pano_size = len(self.pano_dict)
        logger.warning(f"Filter crawling queue({len(lst)}), starting")
        
        params = [(x, y, None, True) for x, y in lst]
        responds = parallel_process(query_pano, params, True, n_jobs=self.n_jobs)
        
        res = []
        for i, node in zip(responds, lst):
            if i is None:
                visited[node] = i
                continue
            if 'pid' in i:
                if i['pid'] not in self.pano_dict:
                    res.append((*node, i['pid']))
                    self.pano_dict[i['pid']] = i['info']
                else:
                    visited[node] = i['pid']
        logger.warning(f"Filter crawling queue: {len(lst)} -> {len(res)}, pano size {ori_pano_size} -> {len(self.pano_dict)}")

        return res


    """" Pano transfer module """
    def pano_dict_to_gdf(self, pano_dict):
        return gpd.GeoDataFrame(pano_dict).T.set_crs(epsg=4326)


    def extract_gdf_roads(self, panos:gpd.GeoDataFrame):
        def _panos_to_line(df):
            res = {
                'RID': df.iloc[0].RID,
                'src': df.iloc[0].name,
                'dst': df.iloc[-1].name,
                'Panos': df[["Order", 'DIR', 'MoveDir',  'Type', 'X', "Y"]].reset_index().to_dict('records')

            }
            
            if df.shape[0] == 1:
                res['geometry'] = LineString([df.iloc[0].geometry.coords[0], df.iloc[0].geometry.coords[0]])
            else:    
                res['geometry'] = LineString([item.geometry.coords[0] for index, item in df.iterrows() ])
            
            return gpd.GeoDataFrame([res])

        panos.sort_values(["RID", "Order"], inplace=True)
        
        return panos.groupby("RID").apply(_panos_to_line).set_index("RID")


    def extract_gdf_panos_from_key_pano(self, panos:gpd.GeoDataFrame, update_dir=True, sim_thred=.15):
        """extract gdf_panos from key pano infomation

        Args:
            panos ([type]): [description]
            update_dir (bool, optional): [description]. Defaults to False.
            sim_thred (float, optional): [description]. Defaults to .15.
        """

        def _cal_dir_diff(x):
            return ((x+360)-180) % 360

        def _extract_pano_helper(item):
            for r in item["Roads"]:
                if r['IsCurrent'] != 1:
                    continue
                
                sorted(r['Panos'], key = lambda x: x['Order'])
                for idx, pano in enumerate(r['Panos']):
                    pano['RID'] = r['ID']
                
                return r['Panos']

            return None

        def _update_key_pano_move_dir(df, gdf_key_panos):
            # TODO 厘清原因
            """Update the moving direction of panos, for the heading of the last point in eahc segment is usually zero. 

            Args:
                gdf (GeoDataFrame): The original panos dataframe
                gdf_key_panos (GeoDataFrame): The key panos dataframe

            Returns:
                [GeoDataFrame]: The panos dataframe after change the moving direaction
            """
            count_dict = df.RID.value_counts().to_dict()

            # the first pano in the road
            con = df.Order==0
            # TODO 补充抓取端点
            df.loc[con, 'MoveDir'] = df[con].apply(lambda x: gdf_key_panos.loc[x.name].MoveDir if x.name in gdf_key_panos.index else 0, axis=1)
            
            con1 = con & df.apply(lambda x: count_dict[x.RID] > 1, axis=1)
            df.loc[con1, 'dir_sim'] = df[con1].apply(lambda x: math.cos(azimuth_diff(x.MoveDir, x.DIR))+1, axis=1)

            # the last pano in the road
            df.sort_values(["RID", 'Order'], ascending=[True, False], inplace=True)
            idx = df.groupby("RID").head(1).index
            df.loc[idx, 'MoveDir'] = df.loc[idx].apply( lambda x: gdf_key_panos.loc[x.name]['MoveDir'] if x.name in gdf_key_panos.index else -1, axis=1 )
            
            return df

        def _update_neg_dir_move_dir(df, sim_thred=.15):
            """增加判断，若是反方向则变更顺序

            Args:
                gdf_panos ([type]): [description]

            Returns:
                [type]: [description]
            """
            df.loc[:, 'revert'] = False
            
            rids = df[df.dir_sim < sim_thred].RID.values
            # update Order
            idxs = df.query(f"RID in @rids").index
            df.loc[idxs, 'revert'] = True
            _max_idx_dict = df.loc[idxs].groupby('RID').Order.max()
            df.loc[idxs, 'Order'] = df.loc[idxs].apply(lambda x: _max_idx_dict.loc[x.RID]-x.Order, axis=1)

            # update MoveDir
            idxs = df.query(f"RID in @rids and not MoveDir>=0").index
            df.loc[idxs, 'MoveDir'] = df.loc[idxs, 'DIR'].apply(_cal_dir_diff)
            
            # # update MoveDir for the positive dir
            # rids = gdf_panos[gdf_panos.dir_sim > 2 - sim_thred].RID.values
            # idxs = gdf_panos.query(f"RID in @rids and not MoveDir>=0").index
            # gdf_panos.loc[idxs, 'MoveDir'] = gdf_panos.loc[idxs, 'DIR']
            
            return df


        if isinstance(panos, dict):
            panos = gpd.GeoDataFrame(panos).T
        assert isinstance(panos, gpd.GeoDataFrame), "Check Input"
        
        df = pd.DataFrame.from_records(
                np.concatenate( panos.apply( lambda x: _extract_pano_helper(x), axis=1 ).values )
            ).drop_duplicates()
        df = gpd.GeoDataFrame(df, geometry=df.apply(lambda x: Point(*bd_mc_to_wgs(x['X'], x['Y'])), axis=1), crs='EPSG:4326')
        df.set_index("PID", inplace=True)


        if update_dir:
            _update_key_pano_move_dir(df, panos)
            _update_neg_dir_move_dir(df, sim_thred)
        
        con = df.MoveDir.isna()
        df.loc[con, 'MoveDir'] = df.loc[con, 'DIR']
        df.MoveDir = df.MoveDir.astype(np.int)
        df.sort_values(["RID", 'Order'], ascending=[True, True], inplace=True)   

        return df


    def extract_gdf_roads_from_key_pano(self, panos:gpd.GeoDataFrame):
        
        def _extract_helper(_roads):
            for road in _roads:
                if road['IsCurrent'] != 1:
                    continue

                # shared memory
                r = deepcopy(road)
                sorted(r['Panos'], key = lambda x: x['Order'])
                r['src'] = r['Panos'][0]['PID']
                r['dst'] = r['Panos'][-1]['PID']
                coords = [ bd_mc_to_wgs(p['X'], p['Y']) for p in r['Panos'] ]
                if len(coords) == 1:
                    coords = coords * 2
                r['geometry'] = LineString(coords)

                return r

            return None
        
        if isinstance(panos, dict):
            panos = gpd.GeoDataFrame(panos).T
        assert isinstance(panos, gpd.GeoDataFrame), "Check Input"
        
        gdf_roads = panos.apply( lambda x: _extract_helper(x.Roads), axis=1, result_type='expand' ).drop_duplicates(['ID', 'src', 'dst'])
        gdf_roads.set_index("ID", inplace=True)

        return gdf_roads


    """ Database related """
    def upload_pano_node(self, gdf:gpd.GeoDataFrame=None):
        # 经过测试，上传的 list 数据会自动序列化
        gdf = gdf if gdf is not None else self.gdf_pano_node
        gdf.loc[:, "ID"] = gdf.index
        
        db_pano_base = load_postgis(self.pano_node_pg)
        ori_size, new_size = db_pano_base.shape[0], gdf.shape[0]
        tmp = gdf.append(db_pano_base).drop_duplicates("ID", keep='first')
        
        if ori_size == tmp.shape[0]:
            return True

        return gdf_to_postgis(tmp, self.pano_node_pg)


    def upload_pano_link(self):
        return gdf_to_postgis(self.gdf_pano_link, 'pano_link')


    @property
    def panos(self):
        if hasattr(self, 'gdf_panos'):
            return self.gdf_panos
        
        self.gdf_panos = self.extract_gdf_panos_from_key_pano(self.gdf_pano_node)
        
        return self.gdf_panos


    def parallel_fetch_img(self, panos=None, overwrite=False):
        panos = self.panos if panos is None else panos
        res = get_pano_img_parallel(panos[['PID', "MoveDir"]].values, overwrite=overwrite)

        return res


    def combine_rids(self):
        uf, df_topo, df_topo_prev = combine_rids(self.gdf_pano_node, self.gdf_pano_link, self.panos, plot=True, logger=logger)
        self.df_trajs = uf.trajs_to_gdf()

        return self.df_trajs



def pano_base_main(project_name, geom=None, bbox=None, rewrite=False, logger=None):
    fn = os.path.join(CACHE_FOLDER, f"pano_dict_{project_name}.pkl")
    if os.path.exists(fn) and not rewrite:
        pano_dict = saver.read(fn)
    else:
        pano_dict = crawl_panos_by_area(bbox=bbox, geom=geom, logger=logger)
        saver.save(pano_dict, fn)
    
    gdf_base = pano_dict_to_gdf(pano_dict)
    # gdf_roads_ = extract_gdf_roads_from_key_pano(pano_dict)
    gdf_panos = extract_gdf_panos_from_key_pano(pano_dict, update_dir=True)
    gdf_roads = extract_gdf_roads(gdf_panos)

    res = { 'pano_dict': pano_dict,
            'gdf_base': gdf_base,
            'gdf_roads': gdf_roads,
            'gdf_panos': gdf_panos,
        }

    return res


#%%
if __name__ == '__main__':
    """futian 核心城区"""
    # futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry

    """ search by road name """
    road_name = '南光高速'
    df, _, _ = get_shp_by_name_with_search_API(road_name)
    df, whole_geom = geom_buffer(df, 50)

    self = PanoCrawler('pcl', geofence=whole_geom, n_jobs=32, load_node=True)
    self._crawl_by_area()

    """ search by area """
    # self = PanoCrawler('pcl', bbox=PCL_BBOX, n_jobs=32, load_node=True)
    # self._crawl_by_area()

    self.combine_rids()
    self.panos 

    cp_imgs_to_dataset(road_name, self.df_trajs, 0, base_folder='/home/pcl/dataset/MultiLaneDataset/motorway')
    # self.upload_pano_node()

    # gdf_to_postgis(self.panos, 'test_0111_panos')
    # gdf_to_postgis(self.gdf_pano_link, 'test_0111_links')


    #%%
    # gdf_to_postgis(self.panos, 'test_0111_panos')
    # gdf_to_postgis(self.gdf_pano_link, 'test_0111_links')

    # res = fetch_pano_img_parallel(self.panos, heading_att = 'MoveDir')

# %%
