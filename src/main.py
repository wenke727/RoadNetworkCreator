#%%
from mapAPI import get_road_shp_by_search_API
from roadNetwork import *
from coord.coord_transfer import *
import matplotlib.pyplot as plt
import urllib
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import coordTransform_py.CoordTransform_utils as ct
import time
from collections import deque
import random
import warnings

warnings.filterwarnings(action="ignore")
from db_process import load_from_DB, store_to_DB, ENGINE

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
Log_request = open( os.path.join(os.path.dirname(__file__), 'log/https.log'), mode='a', buffering=1)


def log_helper(log_file, content):
    log_file.write( f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, {content}\n" )
    return 

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
    global DB_connectors, DB_pano_base, DB_panos, DB_roads

    if links is not None:
        DB_connectors = DB_connectors.append(links, ignore_index=True)

    if  DB_pano_base.shape[0] == 0 or DB_pano_base.query( f" ID ==  '{respond['ID']}' ").shape[0] == 0:
        DB_pano_base = DB_pano_base.append(respond, ignore_index=True)

    DB_panos = DB_panos.append(panos,    ignore_index=True)
    DB_roads = DB_roads.append(cur_road, ignore_index=True)

    # if write_to_db:
    #     # config_db = {"con": ENGINE, 'if_exists':'replace'}
    #     # DB_pano_base.to_postgis( name='pano_base_temp', **config_db )
    #     store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    return True

def recognize_link_position(gdf_pano: gpd.GeoDataFrame, links: gpd.GeoDataFrame):
    """recognize the link near the end port or not

    Args:
        gdf_pano (gpd.GeoDataFrame): [description]
        links (gpd.GeoDataFrame): [description]

    Returns:
        [type]: [description]
    """
    link_to_end_port = links.distance(
        gdf_pano.iloc[-1].geometry) * 110*1000
    link_to_start_port = links.distance(
        gdf_pano.iloc[0].geometry) * 110*1000

    return link_to_end_port <= link_to_start_port

def find_nxt_crawl_point_case_one_pano_return( respond, panos, df_order_coords):
    """当pano返回仅有一个记录的时候，通过相对位置信息来获取新的抓取节点"""
    cur_point =  Point( *bd_mc_to_wgs(*[float(i) for i in respond['crawl_coord'].split(',')], 1))
    cur_crawl_position = np.argmin(df_order_coords.distance(cur_point))
    nxt_crawl_position = np.argmin(df_order_coords.distance(panos.loc[0].geometry))
    nxt_dis = df_order_coords.loc[nxt_crawl_position].dis_cum + 20
    nxt_crawl_position = df_order_coords.query( f"dis_cum >= {nxt_dis}" ).index

    if len(nxt_crawl_position) <= 0 or nxt_crawl_position[0] >=  df_order_coords.shape[0] -1:
        return []
    elif nxt_crawl_position[0] < cur_crawl_position:
        nxt_crawl_position = cur_crawl_position + 3 if cur_crawl_position + 3 < df_order_coords.shape[0]-1 else df_order_coords.shape[0]-1
    else:
        nxt_crawl_position = nxt_crawl_position[0]

    print("find_nxt_crawl_point_case_one_pano_return", cur_crawl_position, " -> ", nxt_crawl_position)
    x, y = df_order_coords.loc[nxt_crawl_position].coords
    return [(x, y)]

def traverse_panos_by_road(df_order_coords):
    #TODO 变成反向遍历，links就可以用上
    queue = deque(list(df_order_coords[['id', 'coords']].values))

    nxt_id = 0
    while queue:
        cur_id, cur_node = queue.popleft()
        if nxt_id > cur_id:
            continue
        print( cur_id, nxt_id, cur_node )

        respond, panos, nxt_maybe = query_pano( *cur_node, visualize = False )
        if len(nxt_maybe) == 0:
            nxt_id += 2
        else:
            nxt_id = np.argmin( df_order_coords.distance(Point( bd_mc_to_wgs(*nxt_maybe[0], factor = 1))) )
        
        time.sleep( 1 )
    return

def query_pano(x=None, y=None, pano_id=None, visualize=False, add_to_DB=True, http_log = True, *args, **kwargs):
    # query pano info by two way: 1) x,y 2) pano id
    res = {'crawl_coord': str(x)+","+str(y) if x is not None else None}
    # print(pano_id, res)

    if pano_id is None:
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
        # for att in ['Roads', 'Links']:
            #     if not isinstance(pano_respond[att], str):
            #         continue
            #     pano_respond[att] = eval(pano_respond[att])
        
        pano_respond['crawl_coord'] = res['res_coord'] = f"{float(pano_respond['X'])/100},{float(pano_respond['Y'])/100}"
        panos, nxt = pano_respond_parser(pano_respond, visualize = visualize, add_to_DB=False, *args, **kwargs)
        
        if panos.iloc[-1].PID == pano_id:
            return pano_respond, panos, nxt

        pano_id = panos.iloc[-1].PID

    def get_pano( pano_id ):
        time.sleep( random.uniform(0.5, 1.5)*2 )
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pano_id}"
        if http_log: log_helper( Log_request, f"query ({pano_id}), {url}")
        request = urllib.request.Request(url, method='GET')
        pano_respond = json.loads(urllib.request.urlopen(request).read())
        return pano_respond['content'][0]

    pano_respond = get_pano( pano_id )
    panos, nxt   = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

    if panos.iloc[-1].PID != pano_id: 
        pano_respond = get_pano( panos.iloc[-1].PID )
        res['crawl_coord'] = ','.join([ str(float(x)/100) for x in [panos.iloc[-1]['X'], panos.iloc[-1]['Y']]])
        panos, nxt = pano_respond_parser({**res, **pano_respond}, visualize=visualize, add_to_DB=add_to_DB, *args, **kwargs)

    return pano_respond, panos, nxt

def pano_respond_parser(respond, add_to_DB, visualize, *args, **kwargs):
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): info return by query_pano_detail_by_coord

    Returns:
        [gpd.GeoDataFrame]: the links
    """
    for att in ["X", "Y"]: respond[att] = float(respond[att])
    respond['geometry'] = Point(bd_mc_to_wgs( respond['X'], respond['Y'], factor=100))
    offset_factor = 2
    roads = pd.DataFrame(respond['Roads']).rename({'ID': "RID"}, axis=1) # TODO road 可能有重复的ID

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
        if False:
            # links.loc[:, 'near_end_point'] = recognize_link_position(panos, links)
            # links.loc[:, 'prev_pano_id']   = links.apply( lambda x: cur_road.PID_end if x.near_end_point else cur_road.PID_start, axis=1)
            # links = links.merge(roads[["RID", "Name", "Width"]], on='RID')
            # nxt_coords = links.query('near_end_point')[['X', 'Y', 'PID']].values.tolist()
            pass

    if len(nxt_coords) == 0 and panos.shape[0] > 1:
        nxt_coords = [[*((panos.iloc[-1][['X', 'Y']] + (panos.iloc[-1][['X', 'Y']] - panos.iloc[-2][['X', 'Y']]) * offset_factor)/100), None]]

    if visualize:
        ax = map_visualize(panos, *args, **kwargs)
        gpd.GeoDataFrame([cur_road]).plot( color='black', ax=ax, label = f"Road ({cur_road.Width/100})")

        if len(nxt_coords) > 0:
            ax.scatter(*ct.bd09_to_wgs84(*bd_mc_to_coord(*nxt_coords[0][:2])),
                    label=f"Next point ({list(nxt_coords[0][:2])})", marker='^', zorder = 2, s=200, color = 'green')
        
        panos.head(1).plot(ax=ax, marker="*", markersize = 300, label = f'Starting Point', color = 'blue', zorder =2)
        
        if 'crawl_coord' in respond and respond['crawl_coord'] is not None:
            # t = respond['crawl_coord'];   print(f"respond[crawl_coord]: {t}")
            crawl_point = ct.bd09_to_wgs84(*bd_mc_to_coord(*[float(i) for i in respond['crawl_coord'].split(',')]))
            ax.scatter(*crawl_point, label = f"Crawl point", color = 'orange', zorder =4, marker = 's' )

        if links.shape[0] > 0:
            # print(f"\n\tlinks type: {type(links)}, shape: {links.shape}")
            links.plot(ax=ax, color='red', zorder=5, label='Links')

        ax.legend(title="Legend", ncol=1, shadow=True)
        title = ax.set_title(f"{cur_road.RID} / {cur_road.PID_start}")

    print(f"\tnxt_coords: { [x[:2] if x[2] is None else x[2] for x in nxt_coords  ]  }")

    if add_to_DB:  
        add_pano_respond_to_DB(respond, panos, links, cur_road)

    return panos, nxt_coords

def intersection_visulize(pano_id=None, visulize=False, *args, **kwargs):
    #  TODO 交叉口的可视化
    pano_respond, panos, nxt = query_pano(pano_id = pano_id, visualize=False, scale =2)

    links = gpd.GeoDataFrame(pano_respond['Links'])
    if links.shape[0] == 0: 
        print('no links')
        return []

    queue, df_panos = [], []
    for pid in links.PID.values:
        res, df_pano, _ =  query_pano(pano_id=pid, add_to_DB=True, visualize=False)
        queue.append(res)
        df_panos.append(df_pano)

    if visulize:
        links.geometry = links.apply( lambda x: LineString( 
            [bd_mc_to_wgs( x.X, x.Y ), bd_mc_to_wgs( x.CPointX, x.CPointY )] ), axis=1 )
        ax = map_visualize(links, 's', **{**kwargs, **{'color':'black'}})

        cur_road = gpd.GeoDataFrame(pano_respond['Roads']).iloc[0]
        # cur_road.plot(color='gray', label='Entrance')
        panos = gpd.GeoDataFrame(cur_road.Panos)
        # panos.loc[:, 'wgs'] = panos.apply(lambda x: bd_mc_to_wgs_vector(x), axis=1)
        panos.geometry = panos.apply(lambda x: Point(*bd_mc_to_wgs_vector(x)), axis=1)
        panos.plot( ax=ax, color='blue', marker = '*', markersize=300, label = 'Pano' )

        
        links.geometry = links.apply( lambda x: Point( bd_mc_to_wgs( x.X, x.Y ) ), axis=1 )
        links.plot(ax=ax, label = 'link point', marker = 'x', markersize = 200, zorder=1)
        
        for df_pano in df_panos:
            df_pano.plot(ax=ax, label = df_pano['RID'].values[0])

        ax.legend(title="Legend", ncol=1, shadow=True)
        plt.axis('off')

    return queue

def traverse_a_intersection():
    # 某一交叉口的点（113.914326,22.726633）
    query_pano(pano_id='09005700121902131627400329U', add_to_DB=True)
    # 四个link
    query_pano(pano_id='09005700121902131201515379U', add_to_DB=True)
    query_pano(pano_id='09005700121902131627380329U', add_to_DB=True)
    query_pano(pano_id='09005700121902131201542409U', add_to_DB=True)
    query_pano(pano_id='09005700121902131203338559U', add_to_DB=True)

def get_road_buffer(road_name, buffer=100):
    df_roads, dirs, ports = get_road_shp_by_search_API(road_name)
    ports = [ [ float(i) for i in  p.split(',')] for p in ports]
    df_copy= df_roads.copy()

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

def isValid_Point(nxt:list, area:Polygon):
    """判断nxt里边的坐标点是否在Polygon里边

    Args:
        nxt (list): [description] e.g.[12681529.64, 2582557.67, '09005700121902131626122949U']
        area (Polygon): 范围

    Returns:
        list: [description]
    """
    df_nxt = gpd.GeoDataFrame([nxt], columns=['x', 'y', 'pano_id'])
    df_nxt.geometry = df_nxt.apply( lambda x: Point( *bd_mc_to_wgs( x.x, x.y, factor=1 ) ), axis=1 )
    return df_nxt.within(area).values[0]

def bfs_helper(x, y, area, pano_id=None, visited=set(), level_max = 200):
    # TODO A* 算法，以方向为导向，优先遍历，但有个问题，就是level就没有用了

    # plot_config = { 'visualize': False, 'add_to_DB': True, 'scale': 2}
    respond, panos, queue = query_pano( x=x, y=y, pano_id = pano_id, http_log=True, visualize = False, add_to_DB=True, scale = 2 )

    level = 0
    while queue and level < level_max:
        nxt_queue = []
        for item in queue:
            if item[2] in visited or not isValid_Point(item, area):
                # print(f"Not Valid {item}")
                continue
            
            res, pa, nxt = query_pano( *item, visualize=False, add_to_DB=True )
            # print(f"\t{item[2]}, len: {len(nxt)}")

            visited.add(item[2])
            nxt_queue += nxt

        print('\n', f"level {level}, queue: {len(queue)}, nxt: {len(nxt_queue)}  ", "=" * 50)
        queue = nxt_queue
        level += 1

        # if level % 5 == 0:
        #     store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    df = DB_roads.query(f"PID_end in {list(visited)} or PID_start in {list(visited)}")
    map_visualize(df, color= 'red')
    return visited

def get_area_and_start_point():
    #! load crawl area
    area     = gpd.read_file('../input/area_test.geojson')
    df_roads = gpd.read_file("../input/深圳市_osm路网_道路.geojson")
    df_nodes = gpd.read_file("../input/深圳市_osm路网_节点.geojson")

    roads = gpd.overlay( df_roads, area, how="intersection" )
    nodes = gpd.overlay( df_nodes, area, how="intersection" )

    if False:
        ax = map_visualize(roads, color='red', scale = 0.1 )
        nodes.plot(ax=ax, )
        ax.axis('off')

        ax = map_visualize( roads.set_geometry( 'start' ), color = 'red', scale = 0.1 )
        ax.axis('off')

    roads.loc[:,'start'] = roads.geometry.apply( lambda i: Point(i.xy[0][0], i.xy[1][0]) )
    roads.loc[:,'end'] = roads.geometry.apply( lambda i: Point(i.xy[0][-1], i.xy[1][-1]) )
    roads.loc[:,'start_bd_mc'] = roads.start.apply(lambda i: wgs_to_bd_mc(*i.coords[0]))
    return roads.start_bd_mc.values.tolist(), area.loc[0].geometry

#%%
if __name__ == "__main__":
    # _ = query_pano(pano_id="09005700011601101430366728N", visualize=True)

    # # 获取单向道路数据
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


    df_roads, ports, road_buffer = get_road_buffer(road_name='南海大道', buffer=200)

    ax = map_visualize( df_roads, color='black' )

    x, y = wgs_to_bd_mc(113.917271,22.503024)
    visited = bfs_helper(x, y, road_buffer, pano_id=None, visited=set(),level_max=400)
    
    # visited = bfs_helper(*ports[0], road_buffer, pano_id=None, visited=set(),level_max=200)
    store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)


    # view intersection
    # pano_respond, panos, nxt = query_pano(pano_id='09005700122003141455078615H', visualize=True)
    # intersection_visulize(pano_respond, visulize=True)
    pass


#%%

# TODO 完善
nxt_panos = intersection_visulize( pano_id='09005700121902131650266199U', visulize=True, scale =8 )



# %%
