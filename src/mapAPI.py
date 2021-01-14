import urllib
import coordTransform_py.CoordTransform_utils as ct
from utils.coord.coord_transfer import bd_coord_to_mc, bd_mc_to_coord, bd_mc_to_wgs
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from PIL import Image
import yaml
import os

config = yaml.load(
    open(os.path.join(os.path.dirname(__file__), 'config.yaml')))
pano_dir = config['data']['pano_dir']
input_dir = config['data']['input_dir']


def get_staticimage(id, heading, folder=pano_dir):
    file_name = f"{folder}/{id}_{heading}.jpg"
    if os.path.exists(file_name):
        return False

    # print(file_name)
    # id = "09005700121902131650290579U"; heading = 87
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={id}&heading={heading}&pitch=0&width=1024&height=768"
    request = urllib.request.Request(url=url, method='GET')
    map = urllib.request.urlopen(request)

    f = open(file_name, 'wb')
    f.write(map.read())
    f.flush()
    f.close()
    return map


def query_pano_detail(pano):
    """
    query the nearby point by a special point id
    @param: static view id
    @return: dataframe
    """
    id = pano['pano_id'] if not isinstance(pano, str) else pano

    url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={id}"
    request = urllib.request.Request(url, method='GET')
    res = json.loads(urllib.request.urlopen(request).read())

    # df = pd.DataFrame( res['content'][0]['Roads'][0]['Panos'] )
    # # df.X, df.Y = df.X/100, df.Y/100
    # # df['lng'] = df.apply( lambda i: MC2LL_lng(i.X, i.Y), axis=1 )
    # # df['lat'] = df.apply( lambda i: MC2LL_lat(i.X, i.Y), axis=1 )
    # # df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*ct.bd09_to_wgs84( i.lng, i.lat )), axis=1 ) )
    # df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*bd_mc_to_wgs( i, ['X', 'Y'] )), axis=1 ) )

    # df.loc[:, 'root'] = id
    return {**pano, **res['content'][0]}


# old version
def query_pano_detail_by_coord(x, y, visualize=False):
    """
    query the nearby point by a special coordination
    @param: x,y
    @return: 
    """
    # x, y = bd_coord_to_mc(x, y)
    # # TODO memo
    # if memo.query( f"crawl_coord == {str(x)+','+str(y)}"):
    #     return memo.query( f"crawl_coord == {str(x)+','+str(y)}")[0]
    info = query_pano_ID_by_coord(x, y)

    if 'pano_id' in info:
        info, df = query_pano_detail(info)
        if visualize:
            map_visualize(df, 'y')
        return info, df, True
    return info, None, False


def query_pano_detail(pano):
    """
    query the nearby point by a special point id
    @param: static view id
    @return: dataframe
    """
    id = pano['pano_id'] if not isinstance(pano, str) else pano

    url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={id}"
    request = urllib.request.Request(url, method='GET')
    res = json.loads(urllib.request.urlopen(request).read())

    # df = pd.DataFrame( res['content'][0]['Roads'][0]['Panos'] )
    # # df.X, df.Y = df.X/100, df.Y/100
    # # df['lng'] = df.apply( lambda i: MC2LL_lng(i.X, i.Y), axis=1 )
    # # df['lat'] = df.apply( lambda i: MC2LL_lat(i.X, i.Y), axis=1 )
    # # df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*ct.bd09_to_wgs84( i.lng, i.lat )), axis=1 ) )
    # df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*bd_mc_to_wgs( i, ['X', 'Y'] )), axis=1 ) )

    # df.loc[:, 'root'] = id
    return {**pano, **res['content'][0]}


def query_pano_ID_by_coord(x, y):
    """Query the the nearest static view ID at (x,y)

    Args:
        x (float): bd lng
        y (float): bd lat

    Returns:
        respond [dict]: pano id, position, status 
    """
    url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
    print(url)
    request = urllib.request.Request(url=url, method='GET')
    res = urllib.request.urlopen(request).read()
    json_data = json.loads(res)

    res = {'crawl_coord': str(x)+","+str(y)}
    if 'content' in json_data:
        res['pano_id'] = json_data['content']['id']
        res['RoadName'] = json_data['content']['RoadName']
        res['res_coord'] = ','.join(
            [str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']]])
    else:
        res['status'] = False

    return res


# def traverse_panos_by_road_old(df_order_coords):
#     #TODO 变成反向遍历，links就可以用上
#     from collections import deque
#     queue = deque(list(df_order_coords[['id', 'coords']].values))

#     nxt_id = 0
#     while queue:
#         cur_id, cur_node = queue.popleft()
#         if nxt_id > cur_id:
#             continue
#         print( cur_id, nxt_id, cur_node )

#         respond, panos, nxt_maybe = query_pano( *cur_node, visualize = False )
#         if len(nxt_maybe) == 0:
#             nxt_id += 2
#         else:
#             nxt_id = np.argmin( df_order_coords.distance(Point( bd_mc_to_wgs(*nxt_maybe[0], factor = 1))) )
        
#         time.sleep( 1 )
#     return

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



if __name__ == '__main__':
    x, y = bd_coord_to_mc(113.950112, 22.545307)
    road_id = query_pano_ID_by_coord(x, y)
    df = query_pano_detail(road_id)
    get_staticimage("09005700121902131650360579U", 76)
    # query_pano_detail_by_coord(12679154.25,2582274.24)

    # for test
    x, y = 12679157.9, 2582278.94
    pano_info = query_pano_ID_by_coord(x, y)
    respond = query_pano_detail(pano_info)
    respond

    pass
