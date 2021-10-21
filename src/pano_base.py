#%%
import os
import math
import json
import time
import random
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import geopandas as gpd
from copy import deepcopy
from collections import deque
from shapely.geometry import Point, LineString, box

from db.db_process import load_postgis, gdf_to_postgis
from utils.df_helper import query_df
from utils.geo_helper import gdf_to_geojson
from utils.http_helper import get_proxy
from utils.log_helper import logbook, LogHelper
from utils.coord.coord_transfer import bd_mc_to_wgs, wgs_to_bd_mc
from utils.geo_plot_helper import map_visualize
from utils.pickle_helper import PickleSaver
from setting import FT_BBOX, PANO_FOLFER

from utils.azimuth_helper import azimuth_diff, azimuth_cos_similarity

saver = PickleSaver()

from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX, LXD_BBOX, CACHE_FOLDER, SZU_BBOX


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
    
    gdf_roads = gdf_panos.apply( lambda x: _extract_helper(x.Roads), axis=1, result_type='expand' ).drop_duplicates(['ID', 'src', 'dst'])
    gdf_roads.set_index("ID", inplace=True)

    return gdf_roads


def extract_gdf_roads(panos):
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


def extract_gdf_panos_from_key_pano(panos, update_dir=False, sim_thred=.15):
    def _extract_helper(item):
        for r in item["Roads"]:
            if r['IsCurrent'] != 1:
                continue
            
            sorted(r['Panos'], key = lambda x: x['Order'])
            for idx, pano in enumerate(r['Panos']):
                pano['RID'] = r['ID']
            
            return r['Panos']

        return None

    def _update_key_pano_move_dir(df, gdf_key_panos):
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
        df.loc[con, 'MoveDir'] = df[con].apply(lambda x: gdf_key_panos.loc[x.name].MoveDir, axis=1)
        
        con1 = con & df.apply(lambda x: count_dict[x.RID] > 1, axis=1)
        df.loc[con1, 'dir_sim'] = df[con1].apply(lambda x: math.cos(azimuth_diff(x.MoveDir, x.DIR)*math.pi/180)+1, axis=1)

        # the last pano in the road
        df.sort_values(["RID", 'Order'], ascending=[True, False], inplace=True)
        idx = df.groupby("RID").head(1).index
        df.loc[idx, 'MoveDir'] = df.loc[idx].apply( lambda x: gdf_key_panos.loc[x.name]['MoveDir'] if x.name in gdf_key_panos.index else -1, axis=1 )
        
        return df

    def _update_neg_dir_move_dir(gdf_panos, sim_thred=.15):
        """增加判断，若是反方向则变更顺序

        Args:
            gdf_panos ([type]): [description]

        Returns:
            [type]: [description]
        """

        def _cal_dir_diff(x):
            return ((x+360)-180) % 360

        rids = gdf_panos[gdf_panos.dir_sim < sim_thred].RID.values
        # update Order
        idxs = gdf_panos.query(f"RID in @rids").index
        gdf_panos.loc[idxs, 'Order'] = -gdf_panos.loc[idxs, 'Order']

        # update MoveDir
        idxs = gdf_panos.query(f"RID in @rids and not MoveDir>=0").index
        gdf_panos.loc[idxs, 'MoveDir'] = gdf_panos.loc[idxs, 'DIR'].apply(_cal_dir_diff)
        
        # # update MoveDir for the positive dir
        # rids = gdf_panos[gdf_panos.dir_sim > 2 - sim_thred].RID.values
        # idxs = gdf_panos.query(f"RID in @rids and not MoveDir>=0").index
        # gdf_panos.loc[idxs, 'MoveDir'] = gdf_panos.loc[idxs, 'DIR']
        
        return gdf_panos


    if isinstance(panos, dict):
        panos = gpd.GeoDataFrame(panos).T
    assert isinstance(panos, gpd.GeoDataFrame), "Check Input"
    
    df = pd.DataFrame.from_records(
            np.concatenate( panos.apply( lambda x: _extract_helper(x), axis=1 ).values )
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


"""traverse relayed functions"""

def query_pano(lon=None, lat=None, pid=None, proxies=True, logger=None, memo=None):
    def _parse_pano_respond(res):
        content = res['content']
        assert len(content) == 1, logger.error(f"check result: {content}")
        
        item = content[0]
        item['geometry'] = Point( bd_mc_to_wgs(item['X'], item['Y']) )
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
                    if logger is not None:
                        logger.warning( f"{url}, error: {respond.status_code}")

                res = json.loads( respond.text )
                if 'content' not in res:
                    if logger is not None:
                        logger.error(f"{url}, Respond: {res}")
                else:
                    if logger is not None:
                        logger.debug(f'{url}, by {proxy}')
                    return res

            except requests.exceptions.RequestException as e:
                logger.error(f"{url}: ", e)

            i += 1
            time.sleep(random.randint(1,10))
        
        return None
    
    url = None
    if lon is not None and lat is not None:
        x, y = wgs_to_bd_mc(lon, lat)
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        res = _request_helper(url)
        
        if 'content' in res and 'id' in res['content']:
            pid = res['content']['id']
            if logger is not None:
                logger.info(f"Coords({lon}, {lat}) -> ({x}, {y}) -> {pid}")
        else:
            if logger is not None:
                logger.warning(f"Coords ({x}, {y}) has no matched pano.")
    
    if pid is not None:
        if memo is not None and pid in memo:
            return {'pid': pid, 'info': memo[pid]}
        
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pid}"
    assert url is not None, "check the input"
    
    res = _request_helper(url)
    pid, item = _parse_pano_respond(res)

    if memo is not None:
        memo[pid] = item
    
    return {'pid': pid, 'info': item}


def query_key_pano(lon=None, lat=None, pid=None, memo={}, logger=None, *args, **kwargs):
    key_panos = []
    respond = query_pano(lon, lat, pid=pid, proxies=True, logger=logger, memo=memo)
    
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
                res = query_pano(pid=p, logger=logger, memo=memo)
                if res is None:
                    continue
            else:
                res = respond
            
            nxt, nxt_record = res['pid'], res['info']
            memo[nxt] = nxt_record

        break
    
    return key_panos


def bfs_panos(pid, bbox=None, geom=None, memo={}, max_layer=500, veobose=True, logger=None):
    if bbox is not None:
        geom = box(*bbox)
    assert geom is not None, "Check Input"
    
    layer = 0
    queue = deque([pid])
    visited = set()
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node in visited:
                continue
            
            key_panos = query_key_pano(pid=node, memo=memo, logger=logger)
            if logger is not None:
                logger.info(f"query_key_pano {pid}: {key_panos}")

            # add nxt pid
            for pid in key_panos:
                if geom is not None and not memo[pid]['geometry'].within(geom):
                    if logger is not None:
                        logger.info(f"node: {pid} not within the geom")
                    continue
                
                for link in memo[pid]['Links']:
                    if link["PID"] in memo:
                        continue
                    
                    queue.append(link["PID"])
                    if logger is not None:
                        logger.debug(f"node: {pid}, links: {[ l['PID'] for l in memo[pid]['Links']]}")
            
            visited.add(node)
            
        if layer > max_layer:
            break

        if veobose:
            print(f"{layer}, len({len(queue)}): {queue}")
        
        layer += 1


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


def crawl_panos_by_area(bbox=None, geom=None, verbose=True, plot=True, logger=None):
    if bbox is not None:
        geom = box(*bbox)
    assert geom is not None, "Check Input"
    
    pano_dict, visited = {}, set()
    queue = deque(get_unvisited_point(gpd.GeoDataFrame(), geom_wkt=geom.wkt, plot=False))
    print(len(queue))
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        
        pid = query_pano(*node, memo=pano_dict, logger=logger)['pid']
        print(node, pid)
        
        origin_size = len(pano_dict)
        bfs_panos(pid, geom=geom, memo=pano_dict, logger=logger)
        visited.add(node)
        
        if len(pano_dict) == origin_size:
            continue
        
        # FIXME: 每次循环计算一次过于费劲，可适当减少节点的数量或者其他策略
        lst = get_unvisited_point(pano_dict_to_gdf(pano_dict), geom_wkt=geom.wkt, plot=False)
        queue = deque([i for i in lst if i not in visited])
        
        if verbose:
            print(len(queue), len(pano_dict))

    if plot:
        gdf_panos = pano_dict_to_gdf(pano_dict)
        gdf_roads = extract_gdf_roads_from_key_pano(gdf_panos)
        map_visualize(gdf_roads)
    
    return pano_dict


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
    logger = LogHelper(log_name='pano.log', stdOutFlag=False).make_logger(level=logbook.INFO)


    """futian 核心城区"""
    futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
    pano_base_res = pano_base_main(project_name="futian", bbox=futian_area.bounds, logger=logger, rewrite=False)
    gdf_base, gdf_roads, gdf_panos = pano_base_res['gdf_base'], pano_base_res['gdf_roads'], pano_base_res['gdf_panos']


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
    pano_dict = crawl_panos_by_area(bbox=FT_BBOX, logger=logger)


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


    """ test_main """
    res = pano_base_main(project_name='lxd', bbox=LXD_BBOX)
    res = pano_base_main(project_name='szu', bbox=SZU_BBOX)


    """ test case 0 """
    # bbox = [114.05014,22.54027, 114.06336,22.54500]
    # pid = '09005700122003271207598823O'

    pid = '09005700121902141247002492D'
    bbox = [114.049865,22.549847, 114.05929,22.55306]
    memo = {}

    bfs_panos(pid, bbox, memo=memo, logger=logger)
    map_visualize(extract_gdf_roads_from_key_pano(memo))

    bbox = [114.049865,22.549847, 114.05929,22.55306]
    pano_dict = crawl_panos_by_area(bbox=bbox, logger=logger)
    map_visualize(extract_gdf_roads_from_key_pano(pano_dict))
