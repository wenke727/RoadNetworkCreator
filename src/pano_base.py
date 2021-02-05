#%%
import os
import matplotlib.pyplot as plt
import urllib
import json
import pandas as pd
import geopandas as gpd
import numpy as np 
import time
import random
import warnings
import seaborn as sns
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon
import coordTransform_py.CoordTransform_utils as ct

from baidu_map import get_road_shp_by_search_API, baidu_API_log
from db.db_process import load_from_DB, store_to_DB, ENGINE
from utils.classes import Digraph
from utils.log_helper import LogHelper, logbook
from utils.utils import load_config
from utils.coord.coord_transfer import *
from utils.geo_plot_helper import map_visualize
warnings.filterwarnings(action="ignore")

config    = load_config()
pano_dir  = config['data']['pano_dir']
input_dir = config['data']['input_dir']
pano_API_log = LogHelper(log_dir=config['data']['log_dir'], log_name='panos_base.log').make_logger(level=logbook.INFO)
DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)


def get_road_buffer(road_name, buffer=100):
    """获取道路的边界线

    Args:
        road_name ([type]): [description]
        buffer (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    df_roads, dirs, ports = get_road_shp_by_search_API(road_name, None)
    ports = [ [ float(i) for i in  p.split(',')] for p in ports]
    df_copy = df_roads.copy()

    df_roads.to_crs(epsg=2384, inplace=True)
    df_roads.loc[:, 'line_buffer'] = df_roads.buffer(buffer)
    df_roads.set_geometry('line_buffer', inplace=True)
    df_roads.to_crs(epsg=4326, inplace=True)

    df_roads.loc[:, 'road'] = 1
    whole_road = df_roads.set_geometry('line_buffer').dissolve(by = 'road')
    whole_road.loc[:, 'geometry'] = whole_road.line_buffer
    whole_road.drop('line_buffer', axis=1,inplace=True )
    whole_road.set_geometry('geometry')
    area = whole_road.iloc[0].geometry

    return df_copy, ports, area


def add_pano_respond_to_DB(respond, panos, links, cur_road, write_to_db = True):
    """insert the record to the dataframe, not the DB

    Args:
        respond (dict): the query API respond 
        panos (gpd.GeoDataFrame): [description]
        links (gpd.GeoDataFrame): [description]
        cur_road (pd.Series): [description]

    Returns:
        Boolean: True
    """
    global DB_connectors, DB_pano_base, DB_panos, DB_roads

    if links is not None:
        DB_connectors = DB_connectors.append(links, ignore_index=True)

    if  DB_pano_base.shape[0] == 0 or DB_pano_base.query( f" ID ==  '{respond['ID']}' ").shape[0] == 0:
        DB_pano_base = DB_pano_base.append(respond, ignore_index=True)

    DB_panos = DB_panos.append(panos,    ignore_index=True)
    DB_roads = DB_roads.append(cur_road, ignore_index=True)

    return True


def query_pano(x=None, y=None, pano_id=None, visualize=False, add_to_DB=True, http_log=True, *args, **kwargs):
    """[summary]

    Args:
        x (float, optional): [description]. Defaults to None.
        y (float, optional): [description]. Defaults to None.
        pano_id (str, optional): [description]. Defaults to None.
        visualize (bool, optional): [description]. Defaults to False.
        add_to_DB (bool, optional): [description]. Defaults to True.
        http_log (bool, optional): Display or not. Defaults to True.

    Returns:
        status (int): -2/-1: error; 1 success, exit record; 2, success, new record
        API respond (dict): the respond of the query
        panos (gpd.GeoDataFrame): the panos in the respond
        nxt (list): the next query point, pid or coordination
        
    """
    global DB_panos, DB_pano_base
    res = {'crawl_coord': str(x)+","+str(y) if x is not None else None}

    # get panoid by coordination
    if pano_id is None:
        url       = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        request   = urllib.request.Request(url=url, method='GET')
        json_data = json.loads(urllib.request.urlopen(request).read())
    
        if 'content' in json_data:
            pano_id = json_data['content']['id']
            res['RoadName'] = json_data['content']['RoadName']
            res['res_coord'] = ','.join([str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']]])
            pano_API_log.info( f"\tcoord ({x}, {y}), {url}")
        else:
            res['status'] = False
            pano_API_log.error(f'\tquery ({x}, {y}) failed, for there is not macthing record')
            return -1, None, None, []
    
    # check the pano id in the DB or not, and the pano is the last point of the segment
    while DB_pano_base.shape[0] > 0 and DB_pano_base.query( f" ID == '{pano_id}' " ).shape[0] > 0:
        pano_respond = DB_pano_base.query( f" ID == '{pano_id}' " ).to_dict('records')[0]
        pano_respond['crawl_coord'] = res['res_coord'] = f"{float(pano_respond['X'])/100},{float(pano_respond['Y'])/100}"
        panos, nxt = pano_respond_parser(pano_respond, visualize = visualize, add_to_DB=False, *args, **kwargs)
        
        if panos.iloc[-1].PID == pano_id:
            return 1, pano_respond, panos, nxt
        pano_id = panos.iloc[-1].PID

    def _get_pano( pano_id, sleep=True ):
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pano_id}"
        request = urllib.request.Request(url, method='GET')
        pano_respond = json.loads(urllib.request.urlopen(request).read())
        if sleep: time.sleep( random.uniform(0.5, 1.5)*2 )
        if http_log:  pano_API_log.info( f"\tpano id: {pano_id}, {url}")

        return pano_respond['content'][0]

    # query pano via Baidu API
    try:
        pano_respond = _get_pano( pano_id )
        panos, nxt   = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

        if panos.iloc[-1].PID != pano_id: 
            pano_respond = _get_pano( panos.iloc[-1].PID )
            res['crawl_coord'] = ','.join([ str(float(x)/100) for x in [panos.iloc[-1]['X'], panos.iloc[-1]['Y']]])
            panos, nxt = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

        return 2, pano_respond, panos, nxt
    except:
        if http_log: pano_API_log.info( f"\tpano id {pano_id}, crawl failed! ")
        
        return -2, None, None, []


def pano_respond_parser(respond, add_to_DB, visualize, console_log=False, *args, **kwargs):
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): Baidu API respond

    Returns:
        [gpd.GeoDataFrame, list]: the links, the next points
    """
    offset_factor = 2

    for att in ["X", "Y"]: respond[att] = float(respond[att])
    respond['geometry'] = Point(bd_mc_to_wgs( respond['X'], respond['Y'], factor=100))
    roads = pd.DataFrame(respond['Roads']).rename({'ID': "RID"}, axis=1) 

    # panos
    cur_road = roads.iloc[0]
    panos = gpd.GeoDataFrame(cur_road.Panos)
    panos.loc[:, "RID"] = respond['RID'] = cur_road.RID
    wgs_coords = panos.apply(lambda x: bd_mc_to_wgs_vector(x), axis=1)
    panos.geometry = wgs_coords.apply(lambda x: Point( *x ))

    # road
    cur_road['PID_start'], cur_road['PID_end'] = panos.PID.values[0], panos.PID.values[-1]
    coords = list(wgs_coords.values)
    cur_road['geometry'] = LineString( coords if len(coords) > 1 else (coords + coords))
    del cur_road['Panos']

    # `nxt_coords`
    links, nxt_coords = pd.DataFrame(), []
    if len(respond['Links']) > 0:
        links = pd.DataFrame(respond['Links'])
        for att in ["X", 'Y']: 
            links.loc[:, att] = links[att].apply( lambda x: float(x) /100 )
        links.loc[:, 'prev_pano_id'] = cur_road.PID_end
        links.loc[:, 'geometry'] = links.apply(lambda i: Point( bd_mc_to_wgs(i.X, i.Y, 1) ), axis=1)
        links = gpd.GeoDataFrame(links, crs='EPSG:4326')
        nxt_coords = links[['X', 'Y', 'PID']].values.tolist()

    if len(nxt_coords) == 0 and panos.shape[0] > 1:
        nxt_coords = [[*((panos.iloc[-1][['X', 'Y']] + (panos.iloc[-1][['X', 'Y']] - panos.iloc[-2][['X', 'Y']]) * offset_factor)/100), None]]

    if visualize:
        fig, ax = map_visualize(panos, *args, **kwargs)
        gpd.GeoDataFrame([cur_road]).plot( color='black', ax=ax, label = f"Road ({cur_road.Width/100})")

        if len(nxt_coords) > 0:
            ax.scatter(*ct.bd09_to_wgs84(*bd_mc_to_coord(*nxt_coords[0][:2])),
                    label=f"Next point ({list(nxt_coords[0][:2])})", marker='^', zorder = 2, s=200, color = 'green')
        
        panos.head(1).plot(ax=ax, marker="*", markersize = 300, label = f'Starting Point', color = 'blue', zorder =2)
        
        if 'crawl_coord' in respond and respond['crawl_coord'] is not None:
            crawl_point = ct.bd09_to_wgs84(*bd_mc_to_coord(*[float(i) for i in respond['crawl_coord'].split(',')]))
            ax.scatter(*crawl_point, label = f"Crawl point", color = 'orange', zorder =4, marker = 's' )

        if links.shape[0] > 0:
            links.plot(ax=ax, color='red', zorder=5, label='Links')

        ax.legend(title="Legend", ncol=1, shadow=True)
        title = ax.set_title(f"{cur_road.RID} / {cur_road.PID_start}")

    if console_log: print(f"\tnxt_coords: { [x[:2] if x[2] is None else x[2] for x in nxt_coords  ]  }")
    if add_to_DB: add_pano_respond_to_DB(respond, panos, links, cur_road)

    return panos, nxt_coords


def isValid_Point(nxt:list, area:Polygon):
    """判断nxt里边的坐标点是否在Polygon里边

    Args:
        nxt (list): the next panos info, e.g.[12681529.64, 2582557.67, '09005700121902131626122949U']
        area (Polygon): the Polygon geometry of the region

    Returns:
        list: [description]
    """
    df_nxt = gpd.GeoDataFrame([nxt], columns=['x', 'y', 'pano_id'])
    df_nxt.geometry = df_nxt.apply( lambda x: Point( *bd_mc_to_wgs( x.x, x.y, factor=1 ) ), axis=1 )
    return df_nxt.within(area).values[0]


def bfs_helper(x, y, area, pano_id=None, max_level=200, visualize=False, console_log=False, log_extra_info=None, auto_save_db=True):
    level     = 0
    query_num = 0
    thres     = 300
    visited   = set()

    _, _, _, queue = query_pano( x=x, y=y, pano_id=pano_id, visualize=False, add_to_DB=True, http_log=True, scale=2 )
    while queue and level < max_level:
        nxt_queue = []
        for item in queue:
            if item[2] in visited or not isValid_Point(item, area):
                continue
            status, res, pa, nxt = query_pano( *item, visualize=False, add_to_DB=True )
            visited.add(item[2])
            nxt_queue += nxt
            
            # auto save db
            if not auto_save_db: continue
            if status == 2: query_num += 1
            if query_num < thres: continue
            store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
            pano_API_log.critical(f"auto save data to postgre database")
            query_num = 0

        if console_log: 
            pano_API_log.info(f"{log_extra_info} level {level}, queue: {len(queue)}, nxt: {len(nxt_queue)} ")
        
        queue = nxt_queue
        level += 1

    if visualize:
        df = DB_roads.query(f"PID_end in {list(visited)} or PID_start in {list(visited)}")
        map_visualize(df, color= 'red')
    
    return visited


def intersection_visulize(pano_id=None, *args, **kwargs):
    """crossing node visulization

    Args:
        pano_id ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    #  交叉口的可视化
    _, pano_respond, panos, nxt = query_pano(pano_id = pano_id, visualize=False, scale =2)

    links = gpd.GeoDataFrame(pano_respond['Links'])
    if links.shape[0] == 0: 
        print('intersection_visulize: no links')
        return [], []

    queue, df_panos, nxt_rids = [], [], []
    for pid in links.PID.values:
        _, res, df_pano, _ =  query_pano(pano_id=pid, add_to_DB=True, visualize=False)
        queue.append(res)
        df_panos.append(df_pano)
        nxt_rids.append( df_pano['RID'].values[0] )

    def draw_panos_as_line(panos, ax, *args, **kwargs):
        coords = panos.geometry.apply( lambda x: x.coords[0] ).values.tolist()
        if len(coords) > 1:
            line = gpd.GeoSeries( LineString( coords ) )
            line.plot( ax=ax, **kwargs )
    
    if True:
        links.geometry = links.apply( lambda x: LineString( 
            [bd_mc_to_wgs( x.X, x.Y ), bd_mc_to_wgs( x.CPointX, x.CPointY )] ), axis=1 )
        fig, ax = map_visualize(links, 's', **{**kwargs, **{'color':'gray'}})

        roads = gpd.GeoDataFrame(pano_respond['Roads'])
        panos = gpd.GeoDataFrame(roads.iloc[0].Panos)
        panos.geometry = panos.apply(lambda x: Point(*bd_mc_to_wgs_vector(x)), axis=1)

        draw_panos_as_line(panos, ax, color='red', label = 'Current road', zorder=2)
        panos[:-1].plot( ax=ax, color='red', zorder = 3 )
        panos[-1:].plot( ax=ax, color='white', edgecolor='red', marker = '*', markersize=300, label = f'Pano ({panos.iloc[-1].PID})', zorder = 3 )

        links.geometry = links.apply( lambda x: Point( bd_mc_to_wgs( x.X, x.Y ) ), axis=1 )
        links.plot(ax=ax, label = 'Link point', marker = 'x', markersize = 200, zorder=1)
        
        colors_range = sns.color_palette('bright',len(df_panos))
        for i, df_pano in  enumerate( df_panos):
            # judge the directions
            linestyle = '-.' if df_pano.iloc[0]['PID'] in links.PID.values else ":"
            
            draw_panos_as_line(df_pano, ax, color=colors_range[i], linestyle=linestyle,  label = df_pano['RID'].values[0])
            df_pano.plot(ax=ax, color=colors_range[i])
            df_pano[-1:].plot(ax=ax, color='white', edgecolor =colors_range[i], zorder=9)

        ax.legend(title="Legend", ncol=1, shadow=True)
        plt.axis('off')

    return queue, nxt_rids


def get_road_origin_points(df_roads):
    """get the origin points, with 0 indegree and more than 1 outdegree, of the roads

    Args:
        df_roads (pd.Datafrem): With attributes `start` and `end`

    Returns:
        origins [list]: The coordinations of origins.
    """
    node_dic = {}
    count = 0

    for i in np.concatenate( [df_roads.start, df_roads.end] ):
        if i in node_dic:
            continue
        node_dic[i] = count
        count += 1

    node = pd.DataFrame([node_dic], index=['id']).T
    edges = df_roads.merge( node, left_on='start', right_index=True ).merge( node, left_on='end', right_index=True, suffixes=['_0', '_1'] )
    node = node.reset_index().rename(columns={"index": 'coord'}).set_index('id')
    
    network = Digraph( edges = edges[['id_0', 'id_1']].values )
    origins = network.get_origin_point()

    return [ [ float(x) for x in node.loc[i, 'coord'].split(",")] for i in  origins]


def traverse_panos_by_road_name(road_name, buffer=300, max_level=300, visualize=True, auto_save_db=True):
    """traverse Baidu panos through the road name. This Function would query the geometry by searching API. Then matching algh is prepared to matched the panos
    to the geometry. 

    Args:
        road_name (str): The name of the road (in Chinese).
        buffer (int, optional): [description]. Defaults to 300, unit: meter.
        max_level (int, optional): [description]. Defaults to 300.
        visualize (bool, optional): [description]. Defaults to True.
        auto_save_db (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    # buffer=500; max_level=400; visualize=True; save=True; road_name = '五和大道'
    df_roads, ports, road_buffer = get_road_buffer(road_name, buffer)
    # map_visualize(df_roads)
    starts  = get_road_origin_points(df_roads)
    visited = set()
    level = 0
    
    try:
        for p in tqdm(starts, desc=road_name):
            log_extra_info = f"{road_name}, {level+1}/{len(starts)}"
            temp = bfs_helper(*p, road_buffer, pano_id=None, max_level=max_level, console_log=True, log_extra_info=log_extra_info, auto_save_db=auto_save_db)
            visited = visited.union(temp)
            level += 1
    except:
        store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
        pano_API_log.error( f"traverse {road_name} error, try to save the records" )

    if visualize:
        df = DB_roads.query(f"PID_end in {list(visited)} or PID_start in {list(visited)}")
        fig, ax = map_visualize(df, color= 'red')
        fig.savefig(os.path.join(config['data']['log_dir'], f"{road_name}.jpg"), pad_inches=0.1, bbox_inches='tight',dpi=600)
        
    store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
    
    return


#%%
if __name__ == "__main__":

    # """ 遍历道路 """
    # lst = ['红荔路', '益田路', '皇岗路', ]
    # lst = ['打石一路', '茶光路',  ]
    # # road_name = "益田路"
    # for road_name in lst:
    #     traverse_panos_by_road_name(road_name, buffer=800, max_level=200)
    
    
    # traverse_panos_by_road_name('打石一路', 50)
    # traverse_panos_by_road_name('新洲路')
    # traverse_panos_by_road_name('益田路')
    # traverse_panos_by_road_name('金田路')
    # traverse_panos_by_road_name('彩田路', 500)
    
    """ 福田区 """
    lst = [
        # '香蜜湖路', 
        '香梅路', '皇岗路', '福田路', '民田路', '福田中心四路', '福田中心五路',  '红树林路',
        '福强路', '福民路', '福华一路', '福中路', '福中一路', '深南中路', '红荔路', '红荔西路', '莲花路', '笋岗西路', '侨香路'
    ]
    
    """ 龙岗区 """
    lst = ['贝尔路','稼先路']
    e_lst = []
    for road_name in lst:
        try:
            traverse_panos_by_road_name(road_name, buffer=500, max_level=200)
        except:
            store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
            e_lst.append(road_name)
    print("error: ", e_lst)
    
    pass



# %%
#%%
starts = starts

df = pd.DataFrame(pd.Series(starts[1:]), columns=['input'])



buffer=500; max_level=400; visualize=True; save=True; road_name = '打石一路'
config = {"area": road_buffer, 'pano_id': None, "max_level": max_level, "console_log":True, "auto_save_db": False}


visited = bfs_helper( *starts[0], **config )




#%%
#! Parrallel

from joblib import Parallel, delayed
import pandas as pd
import multiprocessing as mp

MAX_JOBS = int(mp.cpu_count()) 

def apply_parallel(func, data:pd.DataFrame, params='id', n_jobs = MAX_JOBS, verbose=0, *args, **kwargs):
    if data.shape[0] < n_jobs:
        n_jobs = data.shape[0]
        
    data.loc[:,'group'] = data.index % n_jobs
    df = data.groupby('group')
    
    results = Parallel(
        n_jobs=n_jobs, verbose=verbose)(
            delayed(parallel_helper)(func, group, params, *args, **kwargs) for name, group in df 
        )
    
    print("Done!")
    return results

def parallel_helper(func, data:pd.DataFrame, params, *args, **kwargs):
    res = []
    for index, item in data.iterrows():
        # res.append( item[att] )
        res.append( func( *item[params], *args, **kwargs ))
    
    return res


# parallel_helper(bfs_helper, df[:1], 'input', **config)
res = apply_parallel(bfs_helper, df, 'input', verbose=1,**config)

len(res)

visited = []
for i in res:
    for j in i:
        visited += list(j)

len(set(visited))


# 完善这一块
pd.DataFrame( visited ).drop_duplicates().to_csv("./visited.csv")



# apply_parallel( bfs_helper,  )
# %%

import time
print(time.time())
visited = set()
level = 0

for p in tqdm(starts, desc=road_name):
    log_extra_info = f"{road_name}, {level+1}/{len(starts)}"
    temp = bfs_helper(*p, road_buffer, pano_id=None, max_level=max_level, console_log=True, log_extra_info=log_extra_info, auto_save_db=False)
    visited = visited.union(temp)
    level += 1

print(time.time())
