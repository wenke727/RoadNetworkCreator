import urllib
import coordTransform_py.CoordTransform_utils as ct
from coord.coord_transfer import bd_coord_to_mc, bd_mc_to_coord, bd_mc_to_wgs
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from PIL import Image

PANO_FOLDER = "../output/panos"
DB_pano_base, DB_panos, DB_connectors, DB_roads = pd.DataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()


def get_road_shp_by_search_API(road_name):
    """get road shp by earcing API

    Args:
        road_name ([string]): eg: '光侨路'

    Return:
        gdf, directions, ports
    """

    def points_to_line(line):
        return [ct.bd09_to_wgs84(*bd_mc_to_coord(float(line[i*2]), float(line[i*2+1]))) for i in range(len(line)//2)]

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
    df = gpd.GeoDataFrame(df, geometry=df.apply(
        lambda x: LineString(points_to_line(x.coords.split(','))), axis=1))
    df['start'] = df.apply(lambda x: ','.join(x.coords.split(',')[:2]), axis=1)
    df['end'] = df.apply(lambda x: ','.join(x.coords.split(',')[-2:]), axis=1)
    df.crs = "epsg:4326"
    df.loc[:, 'length'] = df.to_crs('epsg:3395').length
    return df, directions, ports.split(';')


def get_staticimage(id, heading, folder = PANO_FOLDER):
    """
    @desc: get the static image by it's id and heading
    @param: id, panoid
    @param: heading, 0 ~ 360 
    @return: imgae
    """
    # TODO the store form of image
    print(f"id = {id}, heading = {heading}")
    file_name = f"{folder}/{id}.jpg"
    if os.path.exists(file_name):
        # print('file exist')
        return Image.open(file_name)

    # id = "09005700121902131650290579U"; heading = 87
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={id}&heading={heading}&pitch=0&width=1024&height=1024"
    request = urllib.request.Request(url=url, method='GET')
    map = urllib.request.urlopen(request)

    # print(file_name)
    f = open(file_name, 'wb')
    f.write(map.read())
    f.flush()
    f.close()
    return Image.open(file_name)



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


def query_pano(x=None, y=None, panoid=None, visualize=True):
    res = {}
    if panoid is None:
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
        request = urllib.request.Request(url=url, method='GET')
        res = urllib.request.urlopen(request).read()
        json_data = json.loads(res)
    
        res = {'crawl_coord': str(x)+","+str(y)}
        if 'content' in json_data:
            panoid = json_data['content']['id']
            res['pano_id'] = panoid
            res['RoadName'] = json_data['content']['RoadName']
            res['res_coord'] = ','.join([str(float(i)/100) for i in [json_data['content']['x'], json_data['content']['y']]])
        else:
            res['status'] = False
            return None # TODO
    
    global DB_panos
    if DB_panos.shape[0] > 0 and DB_panos.query( f"PID== '{panoid}' " ).shape[0] > 0:
        return None

    url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={panoid}"
    request = urllib.request.Request(url, method='GET')
    pano_respond = json.loads(urllib.request.urlopen(request).read())
    
    nxt = pano_respond_parser({**res, **pano_respond['content'][0]}, add_to_DB=True, visualize=visualize)

    return nxt





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




if __name__ == '__main__':
    x, y = bd_coord_to_mc(113.950112,22.545307)
    road_id = query_pano_ID_by_coord(x, y)
    df = query_pano_detail( road_id )
    get_staticimage("09005700121902131650360579U", 76)
    # query_pano_detail_by_coord(12679154.25,2582274.24)


    # for test
    x, y = 12679157.9, 2582278.94
    pano_info = query_pano_ID_by_coord(x, y)
    respond = query_pano_detail(pano_info)
    respond



    pass








