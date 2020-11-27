import re
import urllib
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import coordTransform_py.CoordTransform_utils as ct
import ctypes 
import os, sys, time
from collections import deque

sys.path.append('/home/pcl/traffic/map_factory')
from PIL import Image
import matplotlib.pyplot as plt

from roadNetwork import *


# module coordination transfer
lib = ctypes.cdll.LoadLibrary( os.path.join(os.path.dirname(__file__), 'baiduCoord.so'))

LL2MC_lng = lib.LL2MC_lng 
LL2MC_lat = lib.LL2MC_lat  
LL2MC_lng.argtypes = [ctypes.c_double, ctypes.c_double]
LL2MC_lat.argtypes = [ctypes.c_double, ctypes.c_double]
LL2MC_lng.restype  = ctypes.c_double
LL2MC_lat.restype  = ctypes.c_double

MC2LL_lat = lib.MC2LL_lat
MC2LL_lng = lib.MC2LL_lng
MC2LL_lat.argtypes = [ctypes.c_double, ctypes.c_double]
MC2LL_lng.argtypes = [ctypes.c_double, ctypes.c_double]
MC2LL_lat.restype  = ctypes.c_double
MC2LL_lng.restype  = ctypes.c_double

def bd_coord_to_mc( lng, lat ):
    return LL2MC_lng(lng, lat), LL2MC_lat(lng, lat)

def bd_mc_to_coord( lng, lat ):
    return MC2LL_lng(lng, lat), MC2LL_lat(lng, lat)

def bd_mc_to_wgs( record, attr = ["X", "Y"], factor = 100 ):
    return ct.bd09_to_wgs84(*bd_mc_to_coord( record[attr[0]]/factor, record[attr[1]]/factor ))


###################################

def get_staticimage(id, heading):
    """
    @desc: get the static image by it's id and heading
    @param: id, panoid
    @param: heading, 0 ~ 360 
    @return: imgae
    """
    # id = "09005700121902131650290579U"; heading = 87
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={id}&heading={heading}&pitch=0&width=1024&height=1024"
    request = urllib.request.Request(url=url, method='GET')
    map = urllib.request.urlopen(request)

    # TODO the store form of image
    file_name = f"{id}.jpg"
    f = open(file_name, 'wb')
    f.write(map.read())
    f.flush()
    f.close()
    return Image.open(file_name)

def get_road_shp_by_search_API(road_name):
    """
    @param: road_name = '光侨路'
    @return gpd.Geodataframe
    """
    def points_to_line(line):
        return [ ct.bd09_to_wgs84( *bd_mc_to_coord(float(line[i*2]), float(line[i*2+1])) ) for i in range(len(line)//2)  ]

    url = f"https://map.baidu.com/?newmap=1&reqflag=pcmap&biz=1&from=webmap&da_par=direct&pcevaname=pc4.1&qt=s&da_src=searchBox.button&wd={urllib.parse.quote(road_name)}&c=340&src=0&wd2=&pn=0&sug=0&l=19&b=(12685428.325,2590847.5;12685565.325,2591337)&from=webmap&sug_forward=&auth=DFK98QE10QLPy1LTFybKvxyESGSRPVGWuxLVLxBVERNtwi04vy77uy1uVt1GgvPUDZYOYIZuVtcvY1SGpuEt2gz4yBWxUuuouK435XwK2vMOuUbNB9AUvhgMZSguxzBEHLNRTVtcEWe1aDYyuVt%40ZPuVteuRtlnDjnCER%40REERG%40EBfiKKvCCu1iifGOb&device_ratio=1&tn=B_NORMAL_MAP&nn=0&u_loc=12684743,2564601&ie=utf-8&t=1606130493139"
    request = urllib.request.Request(url=url, method='GET')
    res = urllib.request.urlopen(request).read()
    json_data = json.loads(res)
    res = pd.DataFrame(json_data['content'])
    # res.query("di_tag == '道路' ")

    # FIXME Maybe the road is not the first record
    lines = json_data['content'][0]['profile_geo']
    directions, ports, lines = lines.split('|')

    df = pd.DataFrame(lines.split(';')[:-1], columns=['coords'])
    df = gpd.GeoDataFrame(df, geometry = df.apply( lambda x: LineString( points_to_line(x.coords.split(',')) ), axis=1 ) )
    df['start'] = df.apply( lambda x: ','.join(x.coords.split(',')[:2]), axis=1 )
    df['end'] = df.apply( lambda x: ','.join(x.coords.split(',')[-2:]), axis=1 )
    df.crs = "epsg:4326"
    df.loc[ :,'length'] = df.to_crs( 'epsg:3395' ).length
    return df, directions, ports.split(';')

def read_Roads(fn = './roads_guangming.geojson'):
    """
    read road from file
    """
    df_roads = gpd.read_file(fn)
    df_roads.query( "name_1 == '光侨路' ", inplace = True )
    map_visualize(df_roads.query( "name_1 == '光侨路' ").head(2))
    return df_roads



# if __name__ == '__main__':
#     x, y = bd_coord_to_mc(113.950112,22.545307)
#     road_id = query_panoID_By_Coord(x, y)
#     df = query_IDs_By_panoID( road_id )
#     map_visualize(df)
#     get_staticimage("09005700121902131650360579U", 76)
#     # query_Pano_IDs_By_Coord(12679154.25,2582274.24)
#     pass    

# 获取单向道路数据
    df_roads, directions, ports = get_road_shp_by_search_API('光侨路')
    # df_roads.to_file('./光侨路.geojson', driver="GeoJSON")
    df_roads = gpd.read_file('../input/光侨路.geojson')
    df_roads_back = df_roads.copy()

    # 东行
    # road_one_side, order = traverse_road( graph['12679023.75,2582246.14'], graph) 
    df_roads = df_roads_back.copy()
    points, graph, zero_indegree_points = extract_roads_info(df_roads)
    # 南行
    res, one_road = traverse_road_consider_reverse_edge( '12685054.46,2591594.75', graph, df_roads )
    map_visualize(one_road, 's')
    one_road.query( "start == '12685797.77,2586241.16' " )
    # one_road.to_file('./光侨路_南行.geojson', driver="GeoJSON")


    # obtain the port of each segment
    lst = list(one_road)
    lst.remove('geometry')
    ports = one_road[ lst ].merge( points[['geometry']], left_on='end', right_index=True )
    ports = gpd.GeoDataFrame(ports, geometry = ports.geometry)
    map_visualize(ports)
    df_order_coords = create_crawl_point(one_road, True)

DB_pano_base, DB_panos, DB_connectors, DB_roads = pd.DataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()


def link_parser(respond):
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): info return by query_Pano_IDs_By_Coord

    Returns:
        [gpd.GeoDataFrame]: the links
    """
    links = pd.DataFrame( respond['Links'] )
    lines = links.apply( lambda link: LineString( [ bd_mc_to_wgs(link, ['CPointX', 'CPointY']), bd_mc_to_wgs(link, ["X", "Y"]) ] ) , axis=1 )
    links = gpd.GeoDataFrame( links, geometry = lines )
    links.loc[:, 'prev_pano_id'] = respond['pano_id']
    return links

def query_panoID_By_Coord(x, y):
    """Query the the nearest static view ID at (x,y)

    Args:
        x (float): bd lng
        y (float): bd lat

    Returns:
        [dict]: 查询的结果, static view ID 
    """

    url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
    request = urllib.request.Request(url = url, method='GET')
    res = urllib.request.urlopen(request).read()
    json_data = json.loads(res)
    
    res = {'crawl_coord': str(x)+","+str(y)}
    if 'content' in json_data:
        res['pano_id'] = json_data['content']['id']
        res['RoadName'] = json_data['content']['RoadName']
        res['res_coord'] = ','.join( [ str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']] ])
    else:
        res['status'] ='error'        

    return res

def query_IDs_By_panoID(pano):
    """
    query the nearby point by a special point id
    @param: static view id
    @return: dataframe
    """
    id = pano['pano_id']
    url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={id}"
    request = urllib.request.Request(url, method='GET')
    res = json.loads(urllib.request.urlopen(request).read())

    df = pd.DataFrame( res['content'][0]['Roads'][0]['Panos'] )
    # df.X, df.Y = df.X/100, df.Y/100
    # df['lng'] = df.apply( lambda i: MC2LL_lng(i.X, i.Y), axis=1 )
    # df['lat'] = df.apply( lambda i: MC2LL_lat(i.X, i.Y), axis=1 )
    # df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*ct.bd09_to_wgs84( i.lng, i.lat )), axis=1 ) )
    df = gpd.GeoDataFrame( df, geometry = df.apply( lambda i: Point(*bd_mc_to_wgs( i, ['X', 'Y'] )), axis=1 ) )

    df.loc[:, 'root'] = id
    return  {**pano, **res['content'][0]}, df

def query_Pano_IDs_By_Coord(x, y, visualize = False):
    """
    query the nearby point by a special coordination
    @param: x,y
    @return: 
    """
    # x, y = bd_coord_to_mc(x, y)
    # # TODO memo
    # if memo.query( f"crawl_coord == {str(x)+','+str(y)}"):
    #     return memo.query( f"crawl_coord == {str(x)+','+str(y)}")[0]
    info = query_panoID_By_Coord(x, y)

    if 'pano_id' in info:
        info, df = query_IDs_By_panoID( info )
        if visualize:
            map_visualize(df, 'y', visualize)
        return info, df
    return info, None

def obtain_panos_info(road_one_way, df_pano = pd.DataFrame(), df_pano_all = gpd.GeoDataFrame() ):
    length = road_one_way.shape[0]

    queue = deque([(0, road_one_way.loc[0, 'coords'])])
    while queue:
        index, node = queue.popleft()
        if index >= length - 1:
            continue    
            
        # x, y = [float(x) for x in node.split(',')]
        x, y = node
        info, pano_record = query_Pano_IDs_By_Coord( x, y, df_pano, False )

        # if df_pano.query( f" crawl_coord == '{info.crawl_coord}' or pano_id == {info.pano_id} " ).shape[0]:
        #     nearest_road_id = index + 1
        # else:
        df_pano = df_pano.append(info, ignore_index=True)
        if pano_record is not None:
            df_pano_all = df_pano_all.append(pano_record, ignore_index=True)
            nxt_road_id = road_one_way.distance( pano_record.iloc[-1].geometry ).nsmallest(1).index[0]
    
            while nxt_road_id <= index:
                nxt_road_id = min(nxt_road_id + 2, length - 1)
        else:
            dis = road_one_way.loc[index, 'dis_cum'] + 20
            ids = road_one_way.query( f"dis_cum > { dis }" ).index
            nxt_road_id = ids[0] if len(ids) > 0 else length - 1
            print( '\tnxt_road_id, ', nxt_road_id )

        # map_visualize( pano_record )
        print(f'id {index} -> {nxt_road_id}, node: {node}')

        queue.append( (nxt_road_id, road_one_way.loc[nxt_road_id, 'coords']) )
        time.sleep(1)
    
    return df_pano, df_pano_all


def parser_pano_respond(respond):    
    """Parse the respond of pano request, and extract links info from it

    Args:
        respond (dict or pd.series): info return by query_Pano_IDs_By_Coord

    Returns:
        [gpd.GeoDataFrame]: the links
    """
    global DB_pano_base, DB_panos, DB_connectors, DB_roads

    # roads
    attrs = ["RID", "Name", "Width"]
    roads = gpd.GeoDataFrame(respond['Roads']).rename({'ID':"RID"}, axis=1)

    # panos
    cur_road = roads.iloc[0]
    panos = gpd.GeoDataFrame( cur_road.Panos )
    panos.loc[:, 'wgs'] = panos.apply(lambda x: bd_mc_to_wgs(x), axis = 1)
    panos.loc[:, "RID"] = cur_road.RID
    panos.geometry = panos.wgs.apply(lambda x: Point(x))

    # road
    cur_road['PID_start'] = panos.PID.values[0]
    cur_road['PID_end'] = panos.PID.values[-1]
    cur_road['geometry'] = LineString( list(panos.wgs.values) )
    del cur_road['Panos']

    # connector
    links = pd.DataFrame( respond['Links'] )
    # lines = links.apply( lambda link: LineString( [ bd_mc_to_wgs(link, ['CPointX', 'CPointY']), bd_mc_to_wgs(link, ["X", "Y"]) ] ) , axis=1 )
    lines = links.apply( lambda link: Point( bd_mc_to_wgs(link, ["X", "Y"]) ) , axis=1 )
    links = gpd.GeoDataFrame( links, geometry = lines )
    links.loc[:, 'prev_pano_id'] = respond['pano_id']
    links = links.merge( roads[attrs], on='RID' )

    DB_pano_base = DB_pano_base.append(respond, ignore_index=True)
    DB_panos = DB_panos.append( panos, ignore_index = True )
    DB_roads = DB_roads.append(cur_road, ignore_index=True)
    DB_connectors = DB_connectors.append(links, ignore_index = True)

    return links

# main

# 周边区域绘制
area = df_roads.buffer(0.0015)
map_visualize( area )


# 异常情况
x, y = 12685067.96,2591582.85
# 正常情况
x, y = 12685486.96,2591170.08



DB_pano_base, DB_panos, DB_connectors, DB_roads = pd.DataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
info, df = query_Pano_IDs_By_Coord(x, y)


nxt_pano = parser_pano_respond(info)
PIDs = list(nxt_pano.PID.values)
# map_visualize(nxt_pano)


count = 0
while len(PIDs) > 0 and count < 5:
    nxt_PIDs = []
    for nxt in PIDs:
        if DB_panos.query( f" PID == '{nxt}' " ).shape[0] > 0:
            continue

        nxt = {'pano_id': nxt}
        info, df = query_IDs_By_panoID(nxt)
        print( nxt )
        # map_visualize( df )
        nxt_pano = parser_pano_respond( info )
        nxt_PIDs += list(nxt_pano.PID.values)
    
    PIDs = nxt_PIDs
    print('PIDs', PIDs)
    count += 1




DB_roads.info()

DB_roads.to_file('DB_roads.geojson', driver="GeoJSON")

DB_connectors.plot()


DB




DB_roads.sort_values( "RID" )


map_visualize(DB_roads)


map_visualize( DB_panos )












links = link_parser(info)
new_id = {'pano_id': links.loc[1,'PID']}
_, gdf_1 = query_IDs_By_panoID(new_id)


# map_visualize(df)
ax = map_visualize(links)
df.plot(ax=ax, color='blue')
gdf.plot(ax=ax, color="green")
gdf_1.plot(ax=ax, color="black")

map_visualize( gdf_1 )



parser_pano_respond( info )

map_visualize(DB_roads)








map_visualize(  )

































# df_pano, df_pano_all = obtain_panos_info(df_order_coords[0:40].reset_index(), df_pano, df_pano_all)
df_pano, df_pano_all = obtain_panos_info(df_order_coords.reset_index(), df_pano, df_pano_all)


df_pano['Rname'].unique()

map_visualize(df_pano_all, 's')


df_pano_all.drop_duplicates()
df_pano_all.to_file('df_pano_all.geojson', driver="GeoJSON")
df_pano.to_hdf('df_nano.h5', key='pano')



df_pano_all = gpd.read_file('./df_pano_all.geojson')
ids = df_pano_all.drop_duplicates().index
df_pano_all[~df_pano_all.index.isin(ids)].root.unique()

df_pano_all[~df_pano_all.index.isin(ids)].plot()
df_pano.query( "pano_id == '01005700001312031243046415T' " )


