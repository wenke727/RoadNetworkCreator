#%%
import math
import json
import time
import random
from pandas.core.indexes.base import Index
import requests
import numpy as np
import pandas as pd
from shapely import geometry
from tqdm import tqdm
import geopandas as gpd
from collections import deque
from shapely.geometry import Point, LineString, box

from db.db_process import load_postgis, gdf_to_postgis
from utils.df_helper import query_df
from utils.log_helper import logbook, LogHelper
from utils.coord.coord_transfer import bd_mc_to_wgs
from utils.geo_plot_helper import map_visualize
from utils.http_helper import get_proxy
from utils.pickle_helper import PickleSaver

saver = PickleSaver()

from setting import SZ_BBOX, GBA_BBOX, PCL_BBOX

g_log_helper = LogHelper(log_name='pano.log', stdOutFlag=False)
logger = g_log_helper.make_logger(level=logbook.INFO)


#%%
"""" Pano traverse module """
def parse_pano_respond(res):
    content = res['content']
    assert len(content) == 1, logger.error(f"check result: {content}")
    
    item = content[0]
    item['geometry'] = Point( bd_mc_to_wgs(item['X'], item['Y']) )
    pid = item['ID']
    del item["ID"]
   
    return pid, item


def query_pano_by_api(x=None, y=None, pid=None, proxies=True, logger=logger):
    url = None
    if x is not None and y is not None:
        url = f'https://mapsv0.bdimg.com/?qt=qsdata&x={x}&y={y}'
    if pid is not None:
        url = f"https://mapsv0.bdimg.com/?qt=sdata&sid={pid}"
    assert url is not None, "check the input"
    
    i = 0 
    while i < 3:   
        try:
            proxy = {'http': get_proxy()} if proxies else None
            respond = requests.get(url, timeout=5, proxies=proxy)
            logger.debug(f'query {url} by {proxy}')
            
            if respond.status_code != 200:
                logger.warning( f"query {url} error: {respond.status_code}")

            res = json.loads( respond.text )
            if 'content' not in res:
                logger.error(f"{pid} url: {url} Respond: {res}")
            else:
                pid, item = parse_pano_respond(res)
                return {'pid': pid, 'info': item}

        except requests.exceptions.RequestException as e:
            logger.error(f"query {url}: ", e)

        i += 1
        time.sleep(random.randint(1,10))

    return None


def query_key_pano(x=None, y=None, pid=None, result={}, logger=logger, *args, **kwargs):
    respond = query_pano_by_api(x, y, pid, logger)
    if respond is None:
        return []

    key_panos = []
    pid, record = respond['pid'], respond['info']
    for road in record['Roads']:
        if road['IsCurrent'] != 1:
            continue
        
        sorted(road['Panos'], key = lambda x: x['Order'])
        panos = [road['Panos'][0]['PID'], road['Panos'][-1]['PID']]
        for p in panos:
            if p in result:
                continue
            
            if p != pid:  
                res = query_pano_by_api(pid=p, logger=logger)
                if res is None:
                    continue
            else:
                res = respond
            
            nxt, nxt_record = res['pid'], res['info']
            result[nxt] = nxt_record
            key_panos.append(nxt)

        break
    
    return key_panos


def extract_gdf_road_from_key_pano(gdf_panos):
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


def extract_gdf_panos_from_key_pano(gdf_panos, update_move_dir=False):
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

    if update_move_dir:
        update_move_dir(df, gdf_panos)

    return df


def update_move_dir(gdf, gdf_key_panos):
    gdf.sort_values(["RID", 'Order'], ascending=[True, False], inplace=True)
    idx = gdf.groupby("RID").head(1).index
    gdf.loc[idx, 'DIR'] = gdf.loc[idx].apply( lambda x: gdf_key_panos.loc[x.name]['MoveDir'], axis=1 )
    gdf.loc[:, 'DIR'] = gdf.loc[:, 'DIR'].astype(np.int)
    gdf.sort_values(["RID", 'Order'], ascending=[True, True], inplace=True)
    
    return gdf


def bfs_panos(pid='09005700121708211337464212S', bbox=None, pano_dict={}, max_layer=50):
    if bbox is not None:
        bbox = box(*bbox)
        
    layer = 0
    queue = deque([pid])
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            key_panos = query_key_pano(pid=node, result=pano_dict)

            # add nxt pid
            for pid in key_panos:
                if bbox is not None and not pano_dict[pid]['geometry'].within(bbox):
                    logger.debug(f"node: {pid} not within the bbox")
                    continue
                
                logger.info(f"node: {pid}, links: {[ l['PID'] for l in pano_dict[pid]['Links']]}")
                for link in pano_dict[pid]['Links']:
                    if link["PID"] in pano_dict:
                        continue
                    queue.append(link["PID"])
            
        if layer > max_layer:
            break

        print(f"{layer}, len({len(queue)}): {queue}")
        layer += 1
        # time.sleep(random.randint(1,10))
        
    return


#%%
# TODO
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


# TODO check with SZU
def get_unvisited_point(road_name = '民治大道', buffer=20):
    # TODO 识别没有抓取到数据的区域
    df_roads, ports, road_buffer = get_road_buffer(road_name, buffer)
    lst = []
    for x in df_roads.geometry.apply( lambda x: x.coords[:] ):
        lst += x

    points = gpd.GeoDataFrame( {'geometry':[ Point(i) for i in  set(lst)]})
    points.loc[:, 'area'] = points.buffer(buffer/110/1000)
    points.reset_index(inplace=True)

    panos = get_features('point', points.total_bounds)
    points.set_geometry('area', inplace=True)

    visited = sorted(gpd.sjoin(left_df=points, right_df=panos, op='contains')['index'].unique().tolist())
    ans = points.query( f"index not in {visited} " )

    return ans 


def get_unvisited_line(road_name='民治大道', buffer=3.75*2.5):
    # TODO 识别没有抓取到数据的区域
    df_roads, ports, road_buffer = get_road_buffer(road_name, buffer)
    df_roads.loc[:, 'area'] = df_roads.buffer(buffer/110/1000)
    df_roads.reset_index(inplace=True)

    # panos = get_features('point', df_roads.total_bounds)
    panos = DB_panos[DB_panos.within(create_polygon_by_bbox(df_roads.total_bounds))]
    df_roads.set_geometry('area', inplace=True)

    visited = sorted(gpd.sjoin(left_df=df_roads, right_df=panos, op='contains')['index'].unique().tolist())
    ans = df_roads.query( f"index not in {visited} " )

    return ans 


#%%
if __name__ == '__main__':
    """ query key pano check """
    tmp_dict = {}
    pid = '09005700121708211232265272S'
    nxt = query_key_pano(pid=pid, result = tmp_dict)

    tmp_dict.keys()

    """ traverse panos in a bbox area"""
    # pid = '09005700121708211337586912S'
    pano_dict = {}
    PCL_BBOX = [113.931914,22.573536, 113.944456,22.580613] #, '09005700121709091541105409Y'
    LXD_BBOX = [113.92423,22.57047, 113.94383,22.58507] #, '09005700121709091541105409Y'
    SZU_BBOX = (113.92370,22.52889,113.94128,22.54281) #,'09005700121708211232265272S'
    FT_BBOX = (114.05097,22.53447,114.05863,22.54605) #, '09005700122003271237204393O'
    # bfs_panos(pid = '09005700122003271237204393O', pano_dict=pano_dict, bbox=FT_BBOX)
    # bfs_panos(pid = '09005700121709091541105409Y', pano_dict=pano_dict, bbox=PCL_BBOX)
    bfs_panos(pid = '09005700121709091541105409Y', pano_dict=pano_dict, bbox=LXD_BBOX)

    # saver.save(pano_dict, '../cache/pano_dict.pkl')


    map_visualize(
        gpd.GeoDataFrame(pano_dict).T
    )

    gdf_panos = gpd.GeoDataFrame(pano_dict).T

    gdf_roads = extract_gdf_road_from_key_pano(gdf_panos)
    gdf_to_postgis(gdf_roads, 'tmp_roads')


    """ extract_gdf_panos_from_key_pano """
    panos_lst = extract_gdf_panos_from_key_pano(gdf_panos)
    update_move_dir(panos_lst, gdf_panos)


# %%
# ! extract topo data
def _parse_road_and_links(pano):
    _roads = pano['Roads']
    
    # judge by `IsCurrent`
    cur_index = 0
    for idx, r in enumerate(_roads):
        if r['IsCurrent'] != 1:
            continue
        cur_index = idx
        break

    nodes = _roads[cur_index]['Panos']

    _rid, src, dst = _roads[cur_index]['ID'], nodes[0]['PID'], nodes[-1]['PID']
    res = [{'rid': _rid, 'src': src, 'dst': dst, 'link':False}]

    links = pano['Links']
    for link in links:
        info = {'src': pano.name, 'dst': link['PID'], 'link':True}
        res.append(info)
        
    return res


def get_topo_from_gdf_pano(gdf_panos, drop_irr_records=True):
    """[summary]

    Args:
        gdf_panos ([type]): [description]
        drop_irr_records (bool, optional): Remove irrelevant records (the origin or destiantion not in the key_panos). Defaults to True.

    Returns:
        [type]: [description]
    """

    topo = []
    for lst in tqdm(gdf_panos.apply(lambda x: _parse_road_and_links(x), axis=1).values):
        topo += lst

    df_topo = pd.DataFrame(topo)
    df_topo.drop_duplicates(df_topo.columns, inplace=True)

    if drop_irr_records:
        pano_ids = gdf_panos.index.values.tolist()
        df_topo.query("src in @pano_ids and dst in @pano_ids", inplace=True)

    # calculate the similarity 
    df_topo.loc[:,['dir_0', 'dir_1']] = df_topo.apply(lambda x: 
        {'dir_0': gdf_panos.loc[x.src]['MoveDir'], 'dir_1': gdf_panos.loc[x.dst]['MoveDir']}, 
        axis=1, result_type='expand')
    df_topo.loc[:, 'similarity'] = df_topo.apply(lambda x: math.cos( azimuth_diff(x.dir_0, x.dir_1) ), axis=1)
    df_topo.loc[~df_topo.link, 'similarity'] = 1

    df_topo.sort_values(['src', 'link', 'similarity'], ascending=[True, True, False], inplace=True)

    df_topo.set_index(['src', 'dst'], inplace=True)

    return df_topo


def azimuth_diff(a, b, unit='radian'):
    """calcaluate the angle diff between two azimuth

    Args:
        a ([type]): Unit: degree
        b ([type]): Unit: degree
        unit(string): radian or degree

    Returns:
        [type]: [description]
    """
    diff = abs(a-b)

    if diff > 180:
        diff = 360-diff

    return diff if unit =='degree' else diff*math.pi/180


pid = '09005700121709091037594139Y'
res = _parse_road_and_links(gdf_panos.loc[pid])

df_topo = get_topo_from_gdf_pano(gdf_panos)
df_topo

# df_roads = df_topo[~df_topo.link]
# df_links = df_topo[df_topo.link]

topo = df_topo.to_dict(orient='index')


# %%

id = '09005700121709091548023739Y'
df_topo.query( "src == @id or dst == @id" )

# %%
graph = {}

for src, dst, link in df_topo.reset_index()[['src', 'dst', 'link']].values:
    graph[src] = graph.get(src, set())
    graph[src].add(dst)

# for s, e, link in df_topo[['src', 'dst', 'link']].values:
#     graph[s] = graph.get(s, dict())
#     graph[s][e] = (link)

graph
#%%

def forward_bfs(node, max_layer=50):
    queue = deque([node])
    path = [node]

    visited_pid = set()
    visited_rid = []

    layer = 1
    while queue:
        for _ in range(len(queue)):
            cur = queue.popleft()
            visited_pid.add(cur)
            
            # for nxt in graph[cur]:
            for nxt, nxt_item in df_topo.loc[cur].iterrows():
                if nxt not in graph:
                    continue
                
                print(layer, (cur, nxt))
                
                if nxt in visited_pid:
                    continue
                
                if topo[(cur, nxt)]['similarity'] < 0.8:
                    break
                
                if not topo[(cur, nxt)]['link']:
                    visited_rid.append(topo[(cur, nxt)]['rid'])
                    if (nxt, nxt) in topo:
                        visited_rid.append(topo[(nxt, nxt)]['rid'])   
                
                path.append(nxt)
                queue.append(nxt)
                
                break
            
        layer += 1
        
        if layer > max_layer:
            break

    return visited_rid


# pid = '09005700121709091547432209Y'
# pid = '09005700121709091539399529Y'
pid = '09005700121709091036077249Y'
pid = '09005700121709031453099872S'
pid = '09005700122003211407076215O'
pid = '09005700122003211407319405O'

rids = forward_bfs(pid)

map_visualize(
    gdf_roads.loc[list(rids)]
)
# %%
query_df(df_topo, 'src', '09005700121709091548023739Y')
















# %%

# ! load `PCL` data
def load_data():
    gdf_pano = load_postgis('panos_lst')
    roads     = load_postgis('roads')

    roads.drop_duplicates('RID', inplace=True)
    roads.set_index('RID', inplace=True)

    attrs = ['Links', 'Roads']
    for i in attrs:
        gdf_pano.loc[:, i] = gdf_pano.apply(lambda x: eval(x[i]), axis=1)

    pd_links_len = gdf_pano.Links.apply(len)
    gdf_pano = gdf_pano[pd_links_len!=0]

    gdf_pano.drop_duplicates('ID', inplace=True)
    gdf_pano.set_index("ID", inplace=True)

    return gdf_pano, roads

# gdf_pano, roads = load_data()
# pano_dict = gdf_pano.to_dict(orient='index')

# %%
def get_pano_links(id, gdf_panos):
    links = gdf_panos.loc[id]['Links']
    
    return links
    
id = '09005700121709091035388809Y'
links = get_pano_links(id, gdf_panos)

queue = deque([])
for link in links:
    node = link['PID']
    if node in gdf_panos.index:
        print(node, 'exist')
        continue
    print(node)
    queue.append(node)


# %%

# TODO forward, backward, 提取topo
def bfs_forward_origin(rid, gdf_pano=gdf_panos, roads=gdf_roads, plot=True, degree_thres=15):
    queue = deque([roads.loc[rid]['PID_end']])
    visited = set()

    path = [rid]

    def _is_valid(cur, nxt):
        return abs(cur['MoveDir'] - nxt['MoveDir']) < degree_thres
    
    while queue:
        cur = queue.pop()
        if cur not in gdf_pano.index:
            continue
        
        cur_info = gdf_pano.loc[cur]

        if cur_info['RID'] in visited:
            continue
        
        for link in cur_info['Links']:
            nxt = link['PID']
            if nxt not in gdf_pano.index:
                print(f"nxt {nxt} not in gdf_pano")
                continue
            
            nxt_info = gdf_pano.loc[nxt]
            
            if not _is_valid(cur_info, nxt_info):
                print(f"{nxt_info['RID']} not valid")
                continue

            nxt_rid = gdf_pano.loc[nxt]['RID']
            nxt_rid_start, nxt_rid_end = roads.loc[nxt_rid][['PID_start','PID_end']]
            
            if link['PID'] != nxt_rid_start:
                print(f"{link['PID']} != {nxt_rid_start}, {link['RID']}")
                continue
            
            print(cur, nxt, cur_info['MoveDir'], nxt_info['MoveDir'], queue )
            queue.append(nxt_rid_end)
            path.append(nxt_info['RID'])

        visited.add(cur_info['RID'])

    if plot:
        roads.loc[path].plot()

    return path

# 创科路北行
rid = 'b1e1bb-bcf9-1d16-0e6d-c02c40'
# 创科路南行
rid = '4bb09e-46d2-40c0-6421-026285'
# 打石一路东行
rid = '81ce8c-d832-1db9-61dc-ee8b61'
# 打石一路西行
rid = 'd39bfb-b6e3-0ad8-582b-b97ccd'

bfs_forward_origin(rid)

