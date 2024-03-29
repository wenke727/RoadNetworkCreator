#%%
import os
import math
import json
import time
import random
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from collections import deque
from shapely.geometry import Point, LineString, box

from db.db_process import load_postgis, gdf_to_postgis
from utils.df_helper import query_df
from utils.http_helper import get_proxy
from utils.log_helper import logbook, LogHelper
from utils.coord.coord_transfer import bd_mc_to_wgs, wgs_to_bd_mc
from utils.geo_plot_helper import map_visualize
from utils.pickle_helper import PickleSaver
from setting import PANO_FOLFER

saver = PickleSaver()

from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX, LXD_BBOX

g_log_helper = LogHelper(log_name='pano.log', stdOutFlag=False)
logger = g_log_helper.make_logger(level=logbook.INFO)


#%%
"""" Pano traverse module """

def pano_dict_to_gdf(pano_dict):
    return gpd.GeoDataFrame(pano_dict).T.set_crs(epsg=4326)


def extract_gdf_roads_from_key_pano(gdf_panos):
    def _extract_helper(_roads):
        for r in _roads:
            if r['IsCurrent'] != 1:
                continue
            
            sorted(r['Panos'], key = lambda x: x['Order'])
            r['src'] = r['Panos'][0]['PID']
            r['dst'] = r['Panos'][-1]['PID']
            coords = [ bd_mc_to_wgs(p['X'], p['Y']) for p in r['Panos'] ]
            if len(coords) == 1:
                coords = coords * 2
            r['geometry'] = LineString(coords)

            return r

        return None
    
    if isinstance(gdf_panos, dict):
        gdf_panos = gpd.GeoDataFrame(gdf_panos).T
    assert isinstance(gdf_panos, gpd.GeoDataFrame), "Check Input"
    
    gdf_roads = gdf_panos.apply( lambda x: _extract_helper(x.Roads), axis=1, result_type='expand' ).drop_duplicates(['ID','src', 'dst'])
    gdf_roads.set_index("ID", inplace=True)

    return gdf_roads


def extract_gdf_panos_from_key_pano(gdf_panos, update_dir=False):
    def _extract_helper(item):
        for r in item["Roads"]:
            if r['IsCurrent'] != 1:
                continue
            
            sorted(r['Panos'], key = lambda x: x['Order'])
            for pano in r['Panos']:
                pano['RID'] = r['ID']
            
            return r['Panos']

        return None

    if isinstance(gdf_panos, dict):
        gdf_panos = gpd.GeoDataFrame(gdf_panos).T
    assert isinstance(gdf_panos, gpd.GeoDataFrame), "Check Input"
    
    df = np.concatenate( gdf_panos.apply( lambda x: _extract_helper(x), axis=1 ).values )
    df = pd.DataFrame.from_records(df).drop_duplicates()
    df = gpd.GeoDataFrame(df, geometry=df.apply(lambda x:  Point(*bd_mc_to_wgs(x['X'], x['Y'])), axis=1), crs='EPSG:4326')
    df.set_index("PID", inplace=True)

    if update_dir:
        update_move_dir(df, gdf_panos)

    return df


def update_move_dir(gdf, gdf_key_panos):
    gdf.sort_values(["RID", 'Order'], ascending=[True, False], inplace=True)
    idx = gdf.groupby("RID").head(1).index
    gdf.loc[idx, 'DIR'] = gdf.loc[idx].apply( lambda x: gdf_key_panos.loc[x.name]['MoveDir'], axis=1 )
    gdf.loc[:, 'DIR'] = gdf.loc[:, 'DIR'].astype(np.int)
    gdf.sort_values(["RID", 'Order'], ascending=[True, True], inplace=True)
    
    return gdf


def parse_pano_respond(res):
    content = res['content']
    assert len(content) == 1, logger.error(f"check result: {content}")
    
    item = content[0]
    item['geometry'] = Point( bd_mc_to_wgs(item['X'], item['Y']) )
    pid = item['ID']
    del item["ID"]
   
    return pid, item


def query_pano_by_api(lon=None, lat=None, pid=None, proxies=True, logger=logger, memo=None):
    
    def __request_helper(url):
        i = 0
        while i < 3:   
            try:
                proxy = {'http': get_proxy()} if proxies else None
                respond = requests.get(url, timeout=5, proxies=proxy)
                logger.debug(f'query {url}, by {proxy}')
                
                if respond.status_code != 200:
                    logger.warning( f"query {url}, error: {respond.status_code}")

                res = json.loads( respond.text )
                if 'content' not in res:
                    logger.error(f"{url}, Respond: {res}")
                else:
                    return res

            except requests.exceptions.RequestException as e:
                logger.error(f"query {url}: ", e)

            i += 1
            time.sleep(random.randint(1,10))
        
        return None
    
    url = None
    if lon is not None and lat is not None:
        x, y = wgs_to_bd_mc(lon, lat)
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        res = __request_helper(url)
        
        if 'content' in res and 'id' in res['content']:
            pid = res['content']['id']
            logger.debug(f"Coords({lon}, {lat}) -> ({x}, {y}) -> {pid}")
        else:
            logger.warning(f"Coords ({x}, {y}) has no mathed pano.")
    
    if pid is not None:
        if memo is not None and pid in memo:
            return {'pid': pid, 'info': memo[pid]}
        
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pid}"
    assert url is not None, "check the input"
    
    res = __request_helper(url)
    pid, item = parse_pano_respond(res)

    if memo is not None:
        memo[pid] = item
    
    return {'pid': pid, 'info': item}


def query_key_pano(lon=None, lat=None, pid=None, memo={}, logger=logger, *args, **kwargs):
    key_panos = []
    respond = query_pano_by_api(lon, lat, pid=pid, logger=logger, memo=memo)
    
    if respond is None:
        return key_panos

    pid, record = respond['pid'], respond['info']
    for road in record['Roads']:
        if road['IsCurrent'] != 1:
            continue
        
        sorted(road['Panos'], key = lambda x: x['Order'])
        panos = [road['Panos'][0]['PID'], road['Panos'][-1]['PID']]
        for p in panos:
            if p in memo:
                continue
            
            if p != pid:  
                res = query_pano_by_api(pid=p, logger=logger, memo=memo)
                if res is None:
                    continue
            else:
                res = respond
            
            nxt, nxt_record = res['pid'], res['info']
            memo[nxt] = nxt_record
            key_panos.append(nxt)

        break
    
    return key_panos


def bfs_panos(pid='09005700121708211337464212S', bbox=None, geom=None, memo={}, max_layer=50, veobose=True):
    if bbox is not None:
        geom = box(*bbox)
    assert geom is not None, "Check Input"
    
    layer = 0
    queue = deque([pid])
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            key_panos = query_key_pano(pid=node, memo=memo)
            logger.debug(f"query_key_pano {pid}: {key_panos}")

            # add nxt pid
            for pid in key_panos:
                if geom is not None and not memo[pid]['geometry'].within(geom):
                    logger.debug(f"node: {pid} not within the geom")
                    continue
                
                logger.info(f"node: {pid}, links: {[ l['PID'] for l in memo[pid]['Links']]}")
                for link in memo[pid]['Links']:
                    if link["PID"] in memo:
                        continue
                    queue.append(link["PID"])
            
        if layer > max_layer:
            break

        if veobose:
            print(f"{layer}, len({len(queue)}): {queue}")
        
        layer += 1
        
    return


def get_unvisited_point(panos, bbox=None, geom_wkt=None, buffer_dis=15, plot=True):
    osm_nodes = load_postgis('topo_osm_shenzhen_node', bbox=bbox, geom_wkt=geom_wkt)

    if panos is None or panos.shape[0]==0:
        return [ (i[0], i[1]) for i in  osm_nodes[['x','y']].values.tolist()]

    osm_nodes.loc[:, 'area'] = osm_nodes.to_crs(epsg=900913).buffer(buffer_dis).to_crs(epsg=4326)
    osm_nodes.reset_index(inplace=True)
    osm_nodes.set_geometry('area', inplace=True)

    visited = gpd.sjoin(left_df=osm_nodes, right_df=panos, op='contains')['index'].unique().tolist()
    res = osm_nodes.query( f"index not in {visited} " )
    
    if plot:
        map_visualize(res)

    return [ (i[0], i[1]) for i in  res[['x','y']].values.tolist()]


def crawl_panos_by_area(bbox=None, geom=None, verbose=True, plot=True):
    if bbox is not None:
        geom = box(*bbox)
    assert geom is not None, "Check Input"
    
    pano_dict, visited = {}, set()
    queue = deque(get_unvisited_point(gpd.GeoDataFrame(), geom_wkt=geom.wkt, plot=False))

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        
        pid = query_pano_by_api(*node, memo=pano_dict)['pid']
        origin_size = len(pano_dict)
        bfs_panos(pid, geom=geom, memo=pano_dict)
        visited.add(node)
        
        if len(pano_dict) == origin_size:
            continue
        
        # TODO: 每次循环计算一次过于费劲，可适当减少节点的数量或者其他策略
        lst = get_unvisited_point(pano_dict_to_gdf(pano_dict), geom_wkt=geom.wkt, plot=False)
        queue = deque([i for i in lst if i not in visited])
        
        if verbose:
            print(len(queue), len(pano_dict))

    if plot:
        gdf_panos = pano_dict_to_gdf(pano_dict)
        gdf_roads = extract_gdf_roads_from_key_pano(gdf_panos)
        map_visualize(gdf_roads)
    
    return pano_dict


#%%
if __name__ == '__main__':
    """ query key pano check """
    # tmp_dict = {}
    # pid = '09005700121708211232265272S'
    # nxt = query_key_pano(pid=pid, result = tmp_dict)
    # tmp_dict.keys()


    """ traverse panos in a bbox area"""
    # pano_dict = {}
    # bfs_panos(pid = '09005700121709091541105409Y', pano_dict=pano_dict, bbox=PCL_BBOX)


    """ crawl_panos_by_area """
    # szu_geom = gpd.read_file('../cache/SZU_geom.geojson').iloc[0].geometry
    # pano_dict = crawl_panos_by_area(geom = szu_geom)
    pano_dict = crawl_panos_by_area(bbox=LXD_BBOX)
    # key_pano_dict = crawl_panos_by_area(box(*PCL_BBOX))


    """ extract data from key panos """
    gdf_key_panos = pano_dict_to_gdf(pano_dict)
    gdf_roads = extract_gdf_roads_from_key_pano(pano_dict)
    gdf_panos = extract_gdf_panos_from_key_pano(pano_dict, update_dir=True)
    map_visualize( gdf_key_panos )


    """ helper """
    saver.save(pano_dict, '../cache/pano_dict_lxd.pkl')
    # gdf_to_postgis(gdf_roads, 'tmp_roads')


    """ SZU check """
    pano_dict = saver.read('../cache/pano_dict_szu.pkl')

# %%

# TODO check pano


def load_exist_pano_imgs(attrs=['pid', 'dir']):
    fn_lst = os.listdir(PANO_FOLFER)
    fn_lst =  pd.DataFrame([ [i] + i.split("_") for i in  fn_lst if 'jpg' in i ], columns=['fn']+attrs)
    fn_lst.loc[:, 'dir'] = fn_lst.dir.apply(lambda x: int(x[:-4]))

    return fn_lst


# %%
def check_unvisited_pano_img(pano_dict):

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

# %%
from pano_img import get_staticimage_batch

new_pano_lst = check_unvisited_pano_img(pano_dict)

get_staticimage_batch(new_pano_lst)

# %%
