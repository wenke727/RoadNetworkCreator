#%%
from numpy.core.numeric import NaN
from mapAPI import get_road_shp_by_search_API
from roadNetwork import *
from coord.coord_transfer import *
import matplotlib.pyplot as plt
import urllib
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import coordTransform_py.CoordTransform_utils as ct
import time
from collections import deque

from db_process import load_from_DB, store_to_DB, ENGINE
# sys.path.append('/home/pcl/traffic/map_factory')


#%%
# DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
# store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)

config_local = {"con": ENGINE, 'if_exists':'replace'}

def add_pano_respond_to_DB(respond, panos, links, cur_road):
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
    DB_pano_base = DB_pano_base.append(respond, ignore_index=True)
    DB_panos = DB_panos.append(panos, ignore_index=True)
    DB_roads = DB_roads.append(cur_road, ignore_index=True)

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

    return link_to_end_port < link_to_start_port

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

def query_pano(x=None, y=None, panoid=None, visualize=True, add_to_DB=False, *args, **kwargs):
    # query pano info by two way: 1) x,y 2) pano id
    res = {}
    if panoid is None:
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        request = urllib.request.Request(url=url, method='GET')
        json_data = json.loads(urllib.request.urlopen(request).read())
    
        res['crawl_coord'] = str(x)+","+str(y)
        if 'content' in json_data:
            panoid = json_data['content']['id']
            res['pano_id'] = panoid
            res['RoadName'] = json_data['content']['RoadName']
            res['res_coord'] = ','.join([str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']]])
        else:
            res['status'] = False
            print(f'query ({x}, {y}) failed, for there is not macthing record')
            return None, None, []
    
    global DB_panos, DB_pano_base
    if DB_panos.shape[0] > 0 and DB_panos.query( f"PID== '{panoid}' " ).shape[0] > 0:
        rid = DB_panos.query( f"PID== '{panoid}' " ).RID.values[0]
        pano_respond =  DB_pano_base.query( f"RID == '{rid}' " ).to_dict('records')[0]

        for att in ['Roads', 'Links']:
            if not isinstance(pano_respond[att], str):
                continue
            pano_respond[att] = eval(pano_respond[att])
        
        pano_respond['crawl_coord'] = res['res_coord'] = f"{float(pano_respond['X'])/100},{float(pano_respond['Y'])/100}"
        panos, nxt = pano_respond_parser(pano_respond, False, visualize, *args, **kwargs)
        # print( f"\tquery pano failed, for {panoid} is existed in the DB, {pano_respond['crawl_coord']}" )
        
        return pano_respond, panos, []

    url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={panoid}"
    request = urllib.request.Request(url, method='GET')
    pano_respond = json.loads(urllib.request.urlopen(request).read())
    pano_respond = {**res, **pano_respond['content'][0]}

    panos, nxt = pano_respond_parser(pano_respond, add_to_DB, visualize, *args, **kwargs)
    return pano_respond, panos, nxt

def pano_respond_parser(respond, add_to_DB, visualize, *args, **kwargs):
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): info return by query_pano_detail_by_coord

    Returns:
        [gpd.GeoDataFrame]: the links
    """
    offset_factor = 2
    attrs = ["RID", "Name", "Width"]
    roads = pd.DataFrame(respond['Roads']).rename({'ID': "RID"}, axis=1) # TODO road 可能有重复的ID

    # panos
    cur_road = roads.iloc[0]
    panos = gpd.GeoDataFrame(cur_road.Panos)
    panos.loc[:, 'wgs'] = panos.apply(lambda x: bd_mc_to_wgs_vector(x), axis=1)
    panos.loc[:, "RID"] = respond['RID'] = cur_road.RID
    panos.geometry = panos.wgs.apply(lambda x: Point(x))

    # road
    cur_road['PID_start'], cur_road['PID_end'] = panos.PID.values[0], panos.PID.values[-1]
    coords = list(panos.wgs.values)
    cur_road['geometry'] = LineString( coords if len(coords) > 1 else (coords + coords))
    del cur_road['Panos']

    # connector
    links, nxt_coords = None, []
    if len(respond['Links']) > 0:
        links = pd.DataFrame(respond['Links'])
        # geometry = links.apply(lambda link: LineString([bd_mc_to_wgs_vector( link, ['CPointX', 'CPointY']), bd_mc_to_wgs_vector(link, ["X", "Y"])]), axis=1)
        geometry = links.apply(lambda link: Point( bd_mc_to_wgs_vector(link, ["X", "Y"]) ), axis=1)
        links = gpd.GeoDataFrame(links, geometry=geometry)
        links.loc[:, 'near_end_point'] = recognize_link_position(panos, links)
        links.loc[:, 'prev_pano_id'] = links.apply(
            lambda x: cur_road.PID_end if x.near_end_point else cur_road.PID_start, axis=1)
        links = links.merge(roads[attrs], on='RID')
        nxt_coords = list(links.query('near_end_point')[['X', 'Y']].values/100)
        

    if len(nxt_coords) == 0 and panos.shape[0] > 1:
        nxt_coords = [(panos.iloc[-1][['X', 'Y']] + (panos.iloc[-1][['X', 'Y']] - panos.iloc[-2][['X', 'Y']]) * offset_factor)/100]

    if add_to_DB: add_pano_respond_to_DB(respond, panos, links, cur_road)

    if visualize:
        ax = map_visualize(panos, *args, **kwargs)
        gpd.GeoDataFrame([cur_road]).plot(
            color='black', ax=ax, label=f"Road ({cur_road.Width/100})")
        if 'crawl_coord' in respond:
            ax.scatter(*ct.bd09_to_wgs84(*bd_mc_to_coord(*[float(i) for i in respond['crawl_coord'].split(',')])), 
                                                     label=f"Crawl point ({respond['crawl_coord']})",
                                                     zorder = 2, color = 'orange')
        if len(nxt_coords) > 0:
            x, y = nxt_coords[0]
            ax.scatter(*ct.bd09_to_wgs84(*bd_mc_to_coord(x, y)),
                       label=f"Next point ({x}, {y})", zorder = 2, color = 'green')
        panos.head(1).plot(ax=ax, marker="*", zorder=1, markersize = 300, label = f'Starting Point', color = 'blue')
        if links is not None:
            links.plot(ax=ax, color='red', linewidth=6,
                       linestyle='--', zorder=3, label='Links')
        ax.legend(title="图例", ncol=1, shadow=True)
        title = ax.set_title(f"{cur_road.RID} / {cur_road.PID_start}")
        print(title)

    return panos, nxt_coords

def intersection_visulize(info, visulize=False,*args, **kwargs):
    #  交叉口的可视化
    links = gpd.GeoDataFrame(info['Links'])
    if links.shape[0] == 0: return []

    queue, df_panos = [], []
    for pid in links.PID.values:
        res, df_pano, _ =  query_pano(panoid=pid, add_to_DB=True, visualize=False)
        queue.append(res)
        df_panos.append(df_pano)

    if visulize:
        links.geometry = links.apply( lambda x: LineString( 
            [bd_mc_to_wgs( x.X, x.Y ), bd_mc_to_wgs( x.CPointX, x.CPointY )] ), axis=1 )
        ax = map_visualize(links, 's', **{**kwargs, **{'color':'black'}})



        cur_road = gpd.GeoDataFrame(info['Roads']).iloc[0]
        # cur_road.plot(color='gray', label='Entrance')
        panos = gpd.GeoDataFrame(cur_road.Panos)
        panos.loc[:, 'wgs'] = panos.apply(lambda x: bd_mc_to_wgs_vector(x), axis=1)
        panos.geometry = panos.wgs.apply(lambda x: Point(x))
        panos.plot( ax=ax, color='blue', marker = '*', markersize=300, label = 'Pano' )

        
        links.geometry = links.apply( lambda x: Point( bd_mc_to_wgs( x.X, x.Y ) ), axis=1 )
        links.plot(ax=ax, label = 'link point', marker = 'x', markersize = 200, zorder=1)
        
        for df_pano in df_panos:
            df_pano.plot(ax=ax, label = df_pano['RID'].values[0])

        ax.legend(title="图例", ncol=1, shadow=True)
        plt.axis('off') # 取消 坐标轴 显示

    if False:
        if 'X' in info:
            ax.scatter(*ct.bd09_to_wgs84(*bd_mc_to_coord(*
                                                [float(info['X']), float(info['Y'])])), 
                                                label=f"Crawl point",
                                                zorder = 2, color = 'orange')

    return queue

def traverse_a_intersection():
    # 某一交叉口的点（113.914326,22.726633）
    query_pano(panoid='09005700121902131627400329U', add_to_DB=True)
    # 四个link
    query_pano(panoid='09005700121902131201515379U', add_to_DB=True)
    query_pano(panoid='09005700121902131627380329U', add_to_DB=True)
    query_pano(panoid='09005700121902131201542409U', add_to_DB=True)
    query_pano(panoid='09005700121902131203338559U', add_to_DB=True)

if __name__ == "__main__":
    # DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB()
    # # 获取单向道路数据
    # # df_roads, directions, ports = get_road_shp_by_search_API('光侨路')
    # # df_roads.to_file('../input/光侨路.geojson', driver="GeoJSON")
    # df_roads = gpd.read_file('../input/光侨路.geojson')
    # df_roads_back = df_roads.copy()
    # # df_roads = df_roads_back.copy()
    # points, graph, zero_indegree_points = extract_roads_info(df_roads)
    # # 南行
    # res, one_road = traverse_road_consider_reverse_edge('12685054.46,2591594.75', graph, df_roads)
    # # 东行
    # res, one_road = traverse_road_consider_reverse_edge('12679023.75,2582246.14', graph, df_roads)
    # map_visualize(one_road, 's', 0.2, (16, 12))
    # # one_road.to_file('../input/光侨路_东行.geojson', driver="GeoJSON")
    # df_order_coords = create_crawl_point(one_road, "line", True)


    # # 发现针对南行情景的效果还是比较差
    # traverse_panos_by_road( df_order_coords )

    # map_visualize(DB_roads)

    # DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new=False)

    # store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads )
    pass

queue = intersection_visulize( info, True, scale = 2 )

#%%
pano_id = '09005700121902170943372965A'
pano_id = '09005700121902131733119505A'
pano_id = '09005700121902171744325875A'
info, temp_pano,_ = query_pano(panoid=pano_id, add_to_DB=True, color = "gray")
queue = intersection_visulize( info, True, scale = 4 )



# # FIXME 交叉口绘制
# # panoid='09005700121902170943361245A'
# panoid = '09005700121902171744325875A'
# info, temp_pano,_ = query_pano(panoid=panoid, add_to_DB=True, visualize=True, lyrs='y', scale = 2)
# queue = intersection_visulize(info, True, scale = 5)



#%%
# pano_id = '09005700121902131633365579U'
pano_id = '09005700121902131631044269U'
info, temp_pano,_ = query_pano(panoid=pano_id, add_to_DB=True, color = "gray")

queue = intersection_visulize( info, True, scale = 2 )

visited = set(info['RID'])
level = 0
while level < 6 and queue:
    nxt_queue = []

    for nxt in queue:
        if nxt['RID'] in visited:
            continue
        nxt_queue += intersection_visulize(nxt, scale = 1 )
        visited.add(nxt['RID'])
        print(f"{level}, {len(queue)} -> {len(nxt_queue)} ({len(visited)}),\t {nxt['RID']}" )
    queue = nxt_queue
    
    ax = map_visualize(DB_roads.query( f"RID in {list(visited)}" ), lyrs='s', scale=1)
    ax.set_title( f'{level} / {len(queue)}' )
    level += 1


map_visualize( DB_roads.query( f"RID in {list(visited)}" ) )

DB_roads.query( f"RID in {list(visited)}" ).to_postgis( 'road_test_1', **config_local  )


rid = 'dbdc68-5892-b68e-5dfe-a778b4'

map_visualize(
    DB_connectors.query( f"RID == '{rid}' " ), lyrs='y'
)

#%%
# TODO 单点的情况，范围很广
info, temp_pano,_ = query_pano(panoid='09005700121902131627400329U', add_to_DB=True)

#%%

map_visualize(DB_roads)

DB_panos.query( "RID == '94148c-159f-3c50-008d-55ec6b' " )


query_pano(panoid='09005700121902131203349179U', scale = 10)




from sqlalchemy import create_engine
STORE_PATH = '/pcl/Data/RoadNetworkCreator/'
ENGINE = create_engine('postgresql://postgres:123456@192.168.135.16:5432/road_network')



DB_roads.to_postgis( name='roads_test', con=ENGINE, if_exists='replace' )


query_pano(panoid='09005700121902121633273745A', lyrs='s', scale=5)


#%%

info, temp_pano,_ = query_pano(panoid="09005700121902131029391275A", add_to_DB=True, scale = 3)

# Roads, Links

# %%


