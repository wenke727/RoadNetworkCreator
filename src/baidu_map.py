#%%
import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import urllib
import json
import coordTransform_py.CoordTransform_utils as ct
import numpy as np
import time 
from tqdm import tqdm

from utils.geo_plot_helper import map_visualize
from utils.utils import load_config
from utils.coord.coord_transfer import bd_mc_to_coord
from utils.log_helper import LogHelper, logbook

import random


config    = load_config()
pano_dir  = config['data']['pano_dir']
input_dir = config['data']['input_dir']

baidu_API_log = LogHelper(log_dir=config['data']['log_dir'], log_name='baidu_map.log').make_logger(level=logbook.INFO)


def get_road_shp_by_search_API(road_name, sleep=False):
    """get road shape by Baidu searching API using the road name

    Args:
        road_name (str): the road name
    """
    def _points_to_line(line):
        return [ct.bd09_to_wgs84(*bd_mc_to_coord(float(line[i*2]), float(line[i*2+1]))) for i in range(len(line)//2)]

    def _searching_api(road_name, log=None):
        url = "https://map.baidu.com/"
        params = {
            'newmap': 1,
            'reqflag': 'pcmap',
            'biz': 1,
            'from': 'webmap',
            'da_par': 'direct',
            'pcevaname': 'pc4.1',
            'qt': 'con',
            'da_src':'searchBox.button',
            'c': 340, # 340,深圳市 Ref: 百度地图城市名称-城市代码（cityCode）关系对照 http://lbsyun.baidu.com/index.php?title=open/dev-res
            'src': 0,
            'wd2': '',
            'pn': 0,
            'sug': 0,
            'l': 19,
            'b': '(12685428.325,2590847.5;12685565.325,2591337)',
            'from': 'webmap',
            'biz_forward':'{%22scaler%22:1,%22styles%22:%22pl%22}',
            'sug_forward': '',
            'auth': 'DFK98QE10QLPy1LTFybKvxyESGSRPVGWuxLVLxBVERNtwi04vy77uy1uVt1GgvPUDZYOYIZuVtcvY1SGpuEt2gz4yBWxUuuouK435XwK2vMOuUbNB9AUvhgMZSguxzBEHLNRTVtcEWe1aDYyuVt%40ZPuVteuRtlnDjnCER%40REERG%40EBfiKKvCCu1iifGOb',
            'device_ratio':1,
            'tn':'B_NORMAL_MAP',
            'nn':0,
            'u_loc':'12684743,2564601',
            'ie':'utf-8',
            }

        params['wd'] = road_name
        
        # query_str = urllib.parse.urlencode(params, timeout=60)
        query_str = urllib.parse.urlencode(params)
        query_str = url+"?"+query_str
        if log is not None:
            log.info( f"{road_name}, query linestring by Baidu API: {query_str}" )
        
        request   = urllib.request.Request(query_str, method='GET')
        res       = urllib.request.urlopen(request).read()
        json_data = json.loads(res)

        return json_data

    fn = os.path.join(input_dir, "road_memo.csv")
    df_roads = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=['name'])
    
    if df_roads.query(f"name == '{road_name}' ").shape[0] > 0:
        json_data = eval( df_roads.query(f"name == '{road_name}' ").respond.values[0] )[0]
    else:
        if sleep: 
            time.sleep(random.triangular(1,15,60))
        # Ref: "../cache/baidu_road_searching_example.json"
        json_data = _searching_api(road_name, baidu_API_log)
        df_roads  = df_roads.append({'name': road_name, 'respond': [json_data]}, ignore_index=True)
        df_roads[['name', 'respond']].to_csv(fn, index=False)

    # FIXME Maybe the road is not the first record
    lines = json_data['content'][0]['profile_geo']
    directions, ports, lines = lines.split('|')

    df = pd.DataFrame(lines.split(';')[:-1], columns=['coords']) # lines[-1] is empty
    df = gpd.GeoDataFrame(df, geometry=df.apply(lambda x: LineString(_points_to_line(x.coords.split(','))), axis=1))
    df['start'] = df.apply(lambda x: ','.join(x.coords.split(',')[:2]), axis=1)
    df['end']   = df.apply(lambda x: ','.join(x.coords.split(',')[-2:]), axis=1)
    df.crs = "epsg:4326"
    df.loc[:, 'length'] = df.to_crs('epsg:3395').length

    return df, directions, ports.split(';')


def roads_from_baidu_search_API(fn=os.path.join(input_dir, "road_memo.csv")):
    """ 从`百度地图`中获取路网 """
    fn = os.path.join(input_dir, "road_memo.csv")
    df_roads = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=['name'])

    records = df_roads.respond.apply(lambda x: pd.DataFrame(eval(x)[0]['content'] if 'content' in eval(x)[0] else []))
    search_respond = pd.concat(records.values)
    search_respond.loc[:, 'catalogID'] = search_respond.loc[:, 'catalogID'].fillna(-1).astype(np.int)
    # FIXME area filter
    con = search_respond.cla.astype(str).str.contains("4, '道路'") & search_respond.area_name.str.contains('深圳')
    search_respond = search_respond[con]
    roads_respond = search_respond.query("road_id == road_id")

    # remove the missing values columns
    roads_respond.dropna(axis=1, how="all", inplace=True)
    roads_respond = roads_respond[~roads_respond.profile_geo.isnull()]
    # roads_respond.drop_duplicates(['name','area'], keep='first', inplace=True)
    roads_respond.reset_index(drop=True, inplace=True)

    roads_respond.loc[:, 'class'] = roads_respond.cla.apply(lambda x: x[-1])
    roads_respond.loc[:, 'directions'] = roads_respond.profile_geo.apply( lambda x: float(x.split("|")[0]))

    # extract road segment
    def _convert_to_lines(content):
        def _points_to_line(line):
            return [ct.bd09_to_wgs84(*bd_mc_to_coord(float(line[i*2]), float(line[i*2+1]))) for i in range(len(line)//2)]

        directions, ports, lines = content.profile_geo.split('|')

        df = pd.DataFrame(lines.split(';')[:-1], columns=['coords'])
        # Six decimal places
        df = gpd.GeoDataFrame(df, geometry=df.apply(
            lambda x: LineString(_points_to_line(x.coords.split(','))), axis=1))

        df.loc[:, 'name'] = content['name']
        df.loc[:, 'primary_uid'] = content['primary_uid']

        return df

    roads = pd.concat(roads_respond.apply(lambda x: _convert_to_lines(x), axis=1).values)

    # move useless attribut
    if True:
        drop_atts = []
        unhashable_atts = []
        for att in list(roads_respond):
            try:
                if roads_respond[att].nunique() == 1:
                    drop_atts.append(att)
            except:
                print(f"{att} unhashable type")
                unhashable_atts.append(att)

        if 'directions' in drop_atts:
            drop_atts.remove('directions')
        drop_atts += ['profile_geo']
        roads_respond.drop(columns=drop_atts, inplace=True)

        for att in unhashable_atts:
            roads_respond.loc[:, att] = roads_respond.loc[:, att].astype(str)
    roads_respond.drop_duplicates('primary_uid', inplace=True)
    roads = roads.merge(roads_respond, on='primary_uid')
    return roads


def fishenet(area, x_step = 0.05, y_step = 0.05, in_sys='wgs84', out_sys='bd09'):
    """create fishnet based on the polygon 

    Args:
        area ([type]): [description]
        x_step (float, optional): [description]. Defaults to 0.05.
        y_step (float, optional): [description]. Defaults to 0.05.
        in_sys (str, optional): [description]. Defaults to 'wgs84'.
        out_sys (str, optional): [description]. Defaults to 'bd09'.

    Returns:
        [type]: [description]
    """
    # x_step = 0.05; y_step = 0.05
    [x0,y0,x1,y1] = area.total_bounds
    net = gpd.GeoDataFrame()
    for x in np.arange( x0, x1, x_step ):
        for y in np.arange( y0, y1, y_step ):
            net = net.append( { 'geometry': Polygon( [(x,y),(x+x_step,y),(x+x_step,y+y_step),(x,y+y_step),(x,y)] )}, ignore_index=True  )
    net = gpd.clip( net, area, keep_geom_type=True)

    if in_sys == out_sys:
        # notice: the coord sequence in bbox
        net.loc[:,'bbox']  = net.bounds.apply( lambda x: [x.miny, x.minx, x.maxy, x.maxx],axis=1 )
        return net

    if in_sys == 'wgs84':
        if out_sys =='bd09':
            bbox = net.bounds.apply( lambda x: [ *ct.wgs84_to_bd09(x.minx, x.miny)[::-1], 
                                                 *ct.wgs84_to_bd09(x.maxx, x.maxy)[::-1]],axis=1 )
        if out_sys =='gcj02':
            bbox = net.bounds.apply( lambda x: [ *ct.wgs84_to_gcj02(x.minx, x.miny)[::-1], 
                                                 *ct.wgs84_to_gcj02(x.maxx, x.maxy)[::-1]],axis=1 )
        
    net.loc[:,'bbox']  = bbox
    return net


def get_roads_name_by_bbox(bounds):
    """get all road names in the district area with the help of Baidu Searcing API
    # TODO 解析detail_info
    Ref: http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-placeapi#service-page-anchor-1-2

    Args:
        bounds ([type]): [description]

    Returns:
        [type]: [description]
    """

    url = 'http://api.map.baidu.com/place/v2/search' 
    params = {
        "query": "道路",
        "region": "深圳",
        "ak": 'VnsZyCwyIC6RNyTK6KanGxZe0UjEiCBP',
        "scope": 2,
        "coord_type":2, # 坐标类型
        "extensions_adcode": "true",
        "output": "json",
        "page_size": 20,
        "page_num": 0
    }
    # 检索矩形区域，多组坐标间以","分隔 `38.76623,116.43213,39.54321,116.46773` lat,lng(左下角坐标),lat,lng(右上角坐标)
    # bounds = "22.516807, 113.992556, 22.552736, 114.00428"
    params["bounds"] = ",".join( map(str, bounds) ) if isinstance(bounds, list) else bounds
    
    df, count = [], 0
    while True:
        query_str = urllib.parse.urlencode(params)
        time.sleep( random.triangular(1, 2, 5) * 3 )
        try:
            request   = urllib.request.Request(url+"?"+query_str, method='GET')
            res       = urllib.request.urlopen(request).read()
            json_data = json.loads(res)
            df.append(pd.DataFrame(json_data['results']))

            baidu_API_log.info( f"[{params['bounds']}], {params['page_num']},  {url}?{query_str}" )
            count += params['page_size']
            
            print( f"{params['page_num']}, {json_data['total']}, {params['bounds']}"  )
            if count >= json_data['total']:
                break
            params['page_num'] += 1
        except:
            print( f"{params['page_num']}, {params['bounds']}"  )
            break
            
    return pd.concat( df ) if len(df) > 0 else None
    
    
def get_roads_name_by_city(area="南山区"):
    """get all road names in the district area with the help of Baidu Searcing API

    Args:
        area (str, optional): [description]. Defaults to "南山区".

    Returns:
        [type]: [description]
    """

    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
    # 南山、罗湖、福田
    area.query(" name=='福田区' or name=='罗湖区' ", inplace=True)
    # fishnet 的间距可能过大, 谨慎使用
    net = fishenet(area, x_step=0.01, y_step=0.01)
    
    res, err = [], []
    for bbox in tqdm( net.bbox.values):
        try:
            df = get_roads_name_by_bbox(bbox)
            if df is not None:
                res.append( df ) 
                print( df.shape[0] )
            time.sleep(random.uniform(20, 60))
        except:
            print( bbox, 'error' )
            err.append(bbox)
    
    for bbox in tqdm(err):
        time.sleep(random.uniform(10, 40)*3)
        try:
            df = get_roads_name_by_bbox(bbox)
            res.append( df ) 
            time.sleep(10)
        except:
            print( bbox, 'error' )
    
    # [ x.shape[0] for x in res ]
    df = pd.concat(res)        
    df.to_csv('road_name.csv')
    
    return df

    
#%%
if __name__ == '__main__':
    road_name = '福中四路'
    df, _, _ = get_road_shp_by_search_API(road_name)


    """" 获取深圳市的道路 """
    df = pd.read_csv('./road_name.csv')
    
    df.loc[:, "detail_info"] = df.detail_info.apply( lambda x: eval(x) )
    df.loc[:, 'tag'] = df.detail_info.apply( lambda x: x['tag'] if 'tag' in x else None)

    road_names = df.query("tag=='道路' and area=='福田区' ").name.values.tolist()

    err = []
    for name in tqdm(road_names):
        print(name)
        try:
            get_road_shp_by_search_API(name, True)
        except:
            err.append(name)
            time.sleep(30)

    
    """ 分析道路的属性 """
    roads = roads_from_baidu_search_API()
    roads_futian = roads.query(" addr.str.contains('福田') ", engine='python')
    map_visualize(roads_futian, lyrs='s')
    list(roads_futian)
    
    roads_futian.catalogID.unique()
    roads_futian.query(f"catalogID == 49").name_x.unique()


#%%
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry

roads_futian = gpd.clip(roads, futian_area)

map_visualize(roads_futian, scale=.05)

#%%
road_name = '滨河大道'
df_, _, _ = get_road_shp_by_search_API(road_name)

map_visualize(df_)

# %%
road_name = '滨河大道辅道'
df_, _, _ = get_road_shp_by_search_API(road_name)

map_visualize(df_)
# %%
