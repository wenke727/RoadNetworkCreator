#%%
import os
from roadNetwork import *
import matplotlib.pyplot as plt
import urllib
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import coordTransform_py.CoordTransform_utils as ct
import time
import random
import warnings
import seaborn as sns
from tqdm import tqdm
warnings.filterwarnings(action="ignore")

from db.db_process import load_from_DB, store_to_DB, ENGINE
from utils.log_helper import log_helper
from utils.utils import load_config
from utils.coord.coord_transfer import *


config    = load_config()
pano_dir  = config['data']['pano_dir']
input_dir = config['data']['input_dir']

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
Log_request = open( os.path.join(config['data']['input_dir'], 'https.log'), mode='a', buffering=1)

#%%

def get_road_shp_by_search_API(road_name):
    """get road shape by Baidu searching API

    Args:
        road_name (str): the road name
    """
    def points_to_line(line):
        return [ct.bd09_to_wgs84(*bd_mc_to_coord(float(line[i*2]), float(line[i*2+1]))) for i in range(len(line)//2)]
    
    fn = os.path.join(input_dir, "road_memo.csv")
    df_roads = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=['name'])

    if df_roads.query(f"name == '{road_name}' ").shape[0] > 0:
        json_data = eval( df_roads.query(f"name == '{road_name}' ").respond.values[0] )[0]
    else:
        url       = f"https://map.baidu.com/?newmap=1&reqflag=pcmap&biz=1&from=webmap&da_par=direct&pcevaname=pc4.1&qt=s&da_src=searchBox.button&wd={urllib.parse.quote(road_name)}&c=340&src=0&wd2=&pn=0&sug=0&l=19&b=(12685428.325,2590847.5;12685565.325,2591337)&from=webmap&sug_forward=&auth=DFK98QE10QLPy1LTFybKvxyESGSRPVGWuxLVLxBVERNtwi04vy77uy1uVt1GgvPUDZYOYIZuVtcvY1SGpuEt2gz4yBWxUuuouK435XwK2vMOuUbNB9AUvhgMZSguxzBEHLNRTVtcEWe1aDYyuVt%40ZPuVteuRtlnDjnCER%40REERG%40EBfiKKvCCu1iifGOb&device_ratio=1&tn=B_NORMAL_MAP&nn=0&u_loc=12684743,2564601&ie=utf-8&t=1606130493139"
        request   = urllib.request.Request(url=url, method='GET')
        res       = urllib.request.urlopen(request).read()
        json_data = json.loads(res)
        df_roads  = df_roads.append({'name': road_name, 'respond': [json_data]}, ignore_index=True)
        df_roads[['name', 'respond']].to_csv(fn, index=False)

    # FIXME Maybe the road is not the first record
    # res = pd.DataFrame(json_data['content']) 
    lines = json_data['content'][0]['profile_geo']
    directions, ports, lines = lines.split('|')

    df = pd.DataFrame(lines.split(';')[:-1], columns=['coords']) # lines[-1] is empty
    df = gpd.GeoDataFrame(df, geometry=df.apply(lambda x: LineString(points_to_line(x.coords.split(','))), axis=1))
    df['start'] = df.apply(lambda x: ','.join(x.coords.split(',')[:2]), axis=1)
    df['end']   = df.apply(lambda x: ','.join(x.coords.split(',')[-2:]), axis=1)
    df.crs = "epsg:4326"
    df.loc[:, 'length'] = df.to_crs('epsg:3395').length

    return df, directions, ports.split(';')


def get_road_buffer(road_name, buffer=100):
    df_roads, dirs, ports = get_road_shp_by_search_API(road_name)
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
    """insert the record to the Database

    Args:
        respond (dict): the query API respond 
        panos (gpd.GeoDataFrame): [description]
        links (gpd.GeoDataFrame): [description]
        cur_road (pd.Series): [description]

    Returns:
        Boolean: True
    """
    # TODO drop_duplicates
    global DB_connectors, DB_pano_base, DB_panos, DB_roads

    if links is not None:
        DB_connectors = DB_connectors.append(links, ignore_index=True)

    if  DB_pano_base.shape[0] == 0 or DB_pano_base.query( f" ID ==  '{respond['ID']}' ").shape[0] == 0:
        DB_pano_base = DB_pano_base.append(respond, ignore_index=True)

    DB_panos = DB_panos.append(panos,    ignore_index=True)
    DB_roads = DB_roads.append(cur_road, ignore_index=True)

    return True


def query_pano(x=None, y=None, pano_id=None, visualize=False, add_to_DB=True, http_log=True, *args, **kwargs):
    res = {'crawl_coord': str(x)+","+str(y) if x is not None else None}
    # print(pano_id, res)

    if pano_id is None:
        # TODO coord_to_pano_memo
        url       = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        request   = urllib.request.Request(url=url, method='GET')
        json_data = json.loads(urllib.request.urlopen(request).read())
        if http_log: log_helper( Log_request, f"query ({x}, {y}), {url}")
    
        if 'content' in json_data:
            pano_id = json_data['content']['id']
            res['RoadName'] = json_data['content']['RoadName']
            res['res_coord'] = ','.join([str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']]])
        else:
            res['status'] = False
            print(f'query ({x}, {y}) failed, for there is not macthing record')
            return None, None, []
    
    global DB_panos, DB_pano_base
    while DB_pano_base.shape[0] > 0 and DB_pano_base.query( f" ID == '{pano_id}' " ).shape[0] > 0:
        pano_respond = DB_pano_base.query( f" ID == '{pano_id}' " ).to_dict('records')[0]
        pano_respond['crawl_coord'] = res['res_coord'] = f"{float(pano_respond['X'])/100},{float(pano_respond['Y'])/100}"
        panos, nxt = pano_respond_parser(pano_respond, visualize = visualize, add_to_DB=False, *args, **kwargs)
        
        if panos.iloc[-1].PID == pano_id:
            return pano_respond, panos, nxt

        pano_id = panos.iloc[-1].PID

    def _get_pano( pano_id, sleep=True ):
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pano_id}"
        if http_log: log_helper( Log_request, f"query ({pano_id}), {url}")
        request = urllib.request.Request(url, method='GET')
        pano_respond = json.loads(urllib.request.urlopen(request).read())
        if sleep: time.sleep( random.uniform(0.5, 1.5)*2 )
        return pano_respond['content'][0]

    pano_respond = _get_pano( pano_id )
    panos, nxt   = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

    if panos.iloc[-1].PID != pano_id: 
        pano_respond = _get_pano( panos.iloc[-1].PID )
        res['crawl_coord'] = ','.join([ str(float(x)/100) for x in [panos.iloc[-1]['X'], panos.iloc[-1]['Y']]])
        panos, nxt = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

    return pano_respond, panos, nxt


def pano_respond_parser(respond, add_to_DB, visualize, console_log=False, *args, **kwargs):
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): info return by query_pano_detail_by_coord

    Returns:
        [gpd.GeoDataFrame]: the links
    """
    for att in ["X", "Y"]: respond[att] = float(respond[att])
    respond['geometry'] = Point(bd_mc_to_wgs( respond['X'], respond['Y'], factor=100))
    offset_factor = 2
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

    links, nxt_coords = pd.DataFrame(), []
    if len(respond['Links']) > 0:
        links = pd.DataFrame(respond['Links'])
        for att in ["X", 'Y']: links.loc[:, att] = links[att].apply( lambda x: float(x) /100 )
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

    if console_log:    
        print(f"\tnxt_coords: { [x[:2] if x[2] is None else x[2] for x in nxt_coords  ]  }")

    if add_to_DB:  
        add_pano_respond_to_DB(respond, panos, links, cur_road)

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


def bfs_helper(x, y, area, pano_id=None, level_max=200, visualize=False, console_log=False):
    # A* 算法，以方向为导向，优先遍历，但有个问题，就是level就没有用了
    # plot_config = { 'visualize': False, 'add_to_DB': True, 'scale': 2}
    respond, panos, queue = query_pano( x=x, y=y, pano_id = pano_id, http_log=True, visualize=False, add_to_DB=True, scale=2 )

    level = 0
    visited = set()
    while queue and level < level_max:
        nxt_queue = []
        for item in queue:
            if item[2] in visited or not isValid_Point(item, area):
                continue
            
            res, pa, nxt = query_pano( *item, visualize=False, add_to_DB=True )
            visited.add(item[2])
            nxt_queue += nxt

        # TODO write to `./log`
        if console_log: print('\n', f"level {level}, queue: {len(queue)}, nxt: {len(nxt_queue)}  ", "=" * 50)
        
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
    pano_respond, panos, nxt = query_pano(pano_id = pano_id, visualize=False, scale =2)

    links = gpd.GeoDataFrame(pano_respond['Links'])
    if links.shape[0] == 0: 
        print('intersection_visulize: no links')
        return [], []

    queue, df_panos, nxt_rids = [], [], []
    for pid in links.PID.values:
        res, df_pano, _ =  query_pano(pano_id=pid, add_to_DB=True, visualize=False)
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


def traverse_panos_by_road(road_name, buffer=500, level_max=400, visualize=True, save=True):
    df_roads, ports, road_buffer = get_road_buffer(road_name, buffer)
    
    starts =  [ [ float(i) for i in x.split(',')] for x in df_roads.start.values.tolist()]
    visited = set()
    count = 0
    
    p = starts[0]
    for p in tqdm(starts):
        temp = bfs_helper(*p, road_buffer, pano_id=None, level_max=level_max, console_log=True)
        visited = visited.union(temp)
        time.sleep(1)

        if len(temp) > 100 or count > 100:
            count = 0
            if save: store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
        else:
            count += len(temp)

    if visualize:
        df = DB_roads.query(f"PID_end in {list(visited)} or PID_start in {list(visited)}")
        map_visualize(df, color= 'red')

    return all


#%%
if __name__ == "__main__":

    """ 遍历道路 """
    # traverse_panos_by_road('沙河西路',buffer=500, level_max=200)
    import pickle
    road_name_lst = pickle.load( open('./road_name_lst_nanshan.pkl', 'rb') )

    failed_record = ['桃园路',
        '明德路',
        '二线巡逻道',
        '科园路',
        '10#桥',
        '深圳灣公路大橋 Shenzhen Bay Bridge',
        '2号路',
        '汕头街',
        '奇趣路',
        '大沙河大桥',
        '兰龙路',
        '格木道',
        '后海立交',
        '兴海大道高架',
        '沙河桥',
        '天宝路',
        '牛罗线',
        '二线关路',
        '前湾一路']
    for i in tqdm(road_name_lst[-185:]):
        try:
            get_road_shp_by_search_API(i)
        except:
            failed_record.append(i)
        time.sleep(10)
    
    traverse_panos_by_road(road_name="望海路", buffer=800, level_max=200)
    store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    """" link 可视化 """
    intersection_visulize(pano_id="09005700011601080935054018N")
    intersection_visulize(pano_id="09005700121709091658098439Y")

    # _ = query_pano(pano_id="09005700011601101430366728N", visualize=True)
    # # df_roads, directions, ports = get_road_shp_by_search_API('光侨路')
    # # df_roads.to_file('../input/光侨路.geojson', driver="GeoJSON")
    # df_roads = gpd.read_file('../input/光侨路.geojson')
    # df_roads_back = df_roads.copy()
    # # df_roads = df_roads_back.copy()
    # points, graph, zero_indegree_points = extract_roads_info(df_roads)
    
    # res, one_road = traverse_road_consider_reverse_edge('12685054.46,2591594.75', graph, df_roads) # 南行
    # # res, one_road = traverse_road_consider_reverse_edge('12679023.75,2582246.14', graph, df_roads) # 东行
    # map_visualize(one_road, 's', 0.2, (16, 12), color='blue')
    # # one_road.to_file('../input/光侨路_东行.geojson', driver="GeoJSON")
    # df_order_coords = create_crawl_point(one_road, "line", True)


    # # # 发现针对南行情景的效果还是比较差
    # # traverse_panos_by_road( df_order_coords )

    # # map_visualize(DB_roads)


    # view intersection
    # pano_respond, panos, nxt = query_pano(pano_id='09005700122003141455078615H', visualize=True)
    # intersection_visulize(pano_respond, visulize=True)
    pass

