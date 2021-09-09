#%%
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from collections import deque
from utils.geo_helper import gdf_to_geojson

from utils.pickle_helper import PickleSaver
from utils.unionFind import UnionFind
from utils.azimuth_helper import azimuth_diff
from utils.geo_plot_helper import map_visualize
from utils.log_helper import LogHelper, logbook
from pano_base import pano_dict_to_gdf, extract_gdf_roads_from_key_pano, extract_gdf_panos_from_key_pano


logger = LogHelper(log_dir="../log", log_name='pano_topo.log').make_logger(level=logbook.INFO)

#%%

def query_edge_by_node(id, df_topo):
    return df_topo.query( "src == @id or dst == @id" )


def plot_node_connections(node, *args, **kwargs):
    """plot node and it's connections

    Args:
        node (str): The node index

    Returns:
        [type]: [description]
    """
    adj_nodes = df_topo.loc[node]
    
    pids = adj_nodes.index.tolist()
    
    fig, ax = map_visualize( gdf_panos.loc[ pids+[node]], color='gray', *args, **kwargs )
    adj_nodes = gdf_panos.merge(adj_nodes, left_index=True, right_index=True).reset_index(drop=False)
    adj_nodes.loc[:, 'info'] = adj_nodes.apply(lambda x: f"{x['index']}, {x.similarity:.3f}",axis=1)
    adj_nodes.sort_values(by='similarity', ascending=False).plot(ax=ax, column='info', legend=True, )
    # adj_nodes.plot(ax=ax, column='similarity', legend=True)
    
    gdf_panos.loc[[node]].plot(ax=ax, marker='*', color='green', markersize=50)
    
    return adj_nodes    


def _parse_road_and_links(pano:dict):
    """Parse the road and links in the pano respond

    Args:
        pano (dict): The pano infomation dict

    Returns:
        list: The list of roads and links related to the pano
    """
    _roads, _links = pano['Roads'], pano['Links']
    
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

    for link in _links:
        info = {'src': pano.name, 'dst': link['PID'], 'link':True}
        res.append(info)
        
    return res


def get_topo_from_gdf_pano(gdf_panos, drop_irr_records=True):
    """Extract topo infomation from panos geodataframe.

    Args:
        gdf_panos (geodataframe): [description]
        drop_irr_records (bool, optional): Remove irrelevant records (the origin or destiantion not in the key_panos). Defaults to True.

    Returns:
        [geodataframe]: [description]
    """

    topo = []
    for lst in tqdm(gdf_panos.apply(lambda x: _parse_road_and_links(x), axis=1).values):
        topo += lst
    df_topo = pd.DataFrame(topo)
    df_topo.drop_duplicates(df_topo.columns, inplace=True)

    # drop irrelevant record
    if drop_irr_records:
        pano_ids = gdf_panos.index.values.tolist()
        df_topo.query("src in @pano_ids and dst in @pano_ids", inplace=True)

    # calculate the similarity 
    df_topo.loc[:,['dir_0', 'dir_1']] = df_topo.apply(
            lambda x: {'dir_0': gdf_panos.loc[x.src]['MoveDir'], 'dir_1': gdf_panos.loc[x.dst]['MoveDir']}, 
            axis=1, 
            result_type='expand'
        )
    df_topo.loc[:, 'similarity'] = df_topo.apply(lambda x: math.cos( azimuth_diff(x.dir_0, x.dir_1) ), axis=1)
    df_topo.loc[~df_topo.link, 'similarity'] = 1

    # compress the rid with one node
    compressed_pids = df_topo.query('dst==src')[['src', 'rid']].set_index('src').to_dict('index')
    df_topo.query('dst != src', inplace=True)
    df_topo.loc[:, 'compressed_rid'] = df_topo.apply(lambda x: compressed_pids[x.src]['rid'] if x.src in compressed_pids else np.nan, axis=1)
    df_topo.sort_values(['src', 'link', 'similarity'], ascending=[True, True, False], inplace=True)
    df_topo.set_index(['src', 'dst'], inplace=True)

    # inverse the graph
    df_topo_prev = df_topo.reset_index().set_index(['dst', 'src'])
    df_topo_prev.sort_values(['dst', 'link', 'similarity'], ascending=[True, True, False], inplace=True)

    return df_topo, df_topo_prev


def build_graph(df_topo):
    """Unfinished

    Args:
        df_topo ([type]): [description]

    Returns:
        [type]: [description]
    """
    graph = {}

    for src, dst, link in df_topo.reset_index()[['src', 'dst', 'link']].values:
        graph[src] = graph.get(src, set())
        graph[src].add(dst)

    return graph


def bfs(node, df_topo, direction=True, visited=set(), plot=True, similar_threds=.9):
    if node not in df_topo.index:
        return []
    
    queue, res = deque([node]), []
    
    while queue:
        cur_pid = queue.popleft()
        if cur_pid not in df_topo.index:
            continue
        logger.info(f"{cur_pid}, {'forword' if direction else 'backward'}: {df_topo.loc[cur_pid].index.values.tolist()}")

        for nxt_pid, nxt in df_topo.loc[cur_pid].iterrows():
            connector = (cur_pid, nxt_pid) if direction else (nxt_pid, cur_pid)
            logger.debug(f"{connector}, {nxt_pid}, {nxt_pid not in df_topo.index}")
            
            if nxt['similarity'] < similar_threds or connector in visited:
                continue
            
            if nxt['compressed_rid'] is not np.nan and nxt['compressed_rid'] not in res:
                res.append(nxt['compressed_rid'])
            if not nxt['link']:
                res.append(nxt['rid'])
            
            queue.append(nxt_pid)
            break

        visited.add(connector)
    
    if plot and len(res) > 0:
        map_visualize( gdf_roads.loc[res] )
    
    return res


def bidirection_bfs(pid, df_topo, df_topo_prev, visited=set(), plot=True):
    """[summary]

    Args:
        pid ([type]): [description]
        df_topo ([type]): [description]
        df_topo_prev ([type]): [description]
        visited_filter (bool, optional): Filter the visited roads. Defaults to True.
        plot (bool, optional): [description]. Defaults to True.
        satellite (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # assert pid in df_topo.index, f"check the pid input: {pid}"
    
    if pid not in df_topo.index and pid not in df_topo_prev.index:
        logger.warning(f"{pid} not in df_topo")
        return []

    rids_0 = bfs(pid, df_topo_prev, False, visited, False)
    rids_1 = bfs(pid, df_topo, True, visited, False)
    rids = rids_0[::-1] + rids_1
   
    if rids and plot:
        roads = gdf_roads.loc[rids]
        roads.loc[:, 'order'] = range(roads.shape[0])
        fig, ax = map_visualize(roads, color='black')
        roads.plot(column='order', legend=True, ax=ax, linewidth=3)
    
    return rids


def combine_rids(gdf_base, gdf_roads, plot=True):
    """Combining rids based on the amuzith similarity.

    Args:
        plot (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    df_topo, df_topo_prev = get_topo_from_gdf_pano(gdf_base)

    res            = {}
    uf             = UnionFind(gdf_roads.index.tolist())
    rid_2_start    = {}
    # 起点和终点都是一样的路段不适宜作为遍历的起始点
    queue          = deque(gdf_roads.query('src != dst').index.tolist())
    visited        = set() # {(src, dst)}
    dulplicate_src = []
    
    while queue:
        cur = gdf_roads.loc[queue.popleft()]
        cur_pid = cur['src']

        if (cur_pid, cur['dst']) in visited:
            logger.debug(f"skip {cur.name}")
            continue
        
        rids = bidirection_bfs(cur_pid, df_topo, df_topo_prev, visited, plot=False)
        if not rids:
            continue

        # origin
        for i in range(len(rids)-1):
            uf.connect(rids[i], rids[i+1])
        # new
        for i in rids:
            if i in rid_2_start:
                logger.warning(f"{i} exist in rid_2_start")
            rid_2_start[i] = rid_2_start.get(i, set())
            rid_2_start[i].add(rids[0])
        
        if rids[0] not in res:
            res[rids[0]] = [rids]
        else:
            res[rids[0]].append(rids)
            dulplicate_src.append(rids[0])
            print(f"dulplicate: {rids[0]}")
        
        logger.info(f"visited {cur_pid}/{cur.name}, links: {rids}")

    if plot:
        fig, ax = map_visualize(gdf_roads, scale=.05)
        gdf_roads.merge( pd.DataFrame(visited, columns=['src', 'dst']), on=['src', 'dst']).plot(ax=ax)

    return res, uf, rid_2_start


def debug_shrink_links():
    # TODO 精简topo
    df = df_topo.reset_index()

    roads = df[~df.rid.isna()]
    links = df[df.rid.isna()]


    srcs = roads.src.unique().tolist()
    dsts = roads.dst.unique().tolist()


    link_lst = links.query(f"src in @dsts")[['src', 'dst']].values.tolist()
    link_set = set( [(src, dst)  for src, dst in link_lst] +
                    [(dst, src)  for src, dst in link_lst]
                )

    links.loc[:, '_filter'] = links.apply(lambda x: (x.src, x.dst) in link_set, axis=1)


    links.query(f'_filter')

    links.query(f'not _filter')

    return


def get_panos_by_rids(rids, gdf_roads, gdf_panos=None, plot=False):
    lst = []

    df = gdf_roads.loc[rids]
    for sub in df.Panos.apply(lambda x: [ i['PID'] for i in x]).values.tolist():
        lst += sub
    
    if plot and gdf_panos is not None:
        map_visualize(gdf_panos.loc[lst])
    
    gdf = gdf_panos.loc[lst]
    # gdf.loc[:, 'PID'] = gdf.index
    
    return gdf.reset_index()


def get_trajectory_by_rid(rid, uf, traj_rid_lst, gdf_roads, plot=True):
    """选择 rid 所在的轨迹，也可以进行可视化

    Args:
        rid ([type]): [description]
        plot (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if rid not in uf.father:
        return []
    
    rids = traj_rid_lst[uf.find(rid)]
    for values in rids:
        if rid in values:
            if plot:
                roads = gdf_roads.loc[values]
                roads.loc[:, 'order'] = range(roads.shape[0])
                fig, ax = map_visualize(roads, color='black')
                roads.plot(column='order', legend=True, ax=ax, linewidth=3)
                
            return values
    
    return []


#%%
if __name__ == '__main__':
    pickler   = PickleSaver()
    # pano_dict = pickler.read('../cache/pano_dict_lxd.pkl')
    pano_dict = pickler.read('../cache/pano_dict_futian.pkl')
    gdf_base = pano_dict_to_gdf(pano_dict)
    gdf_panos = extract_gdf_panos_from_key_pano(gdf_base, update_dir=True)
    gdf_roads = extract_gdf_roads_from_key_pano(gdf_base)
    df_topo, df_topo_prev = get_topo_from_gdf_pano(gdf_base)


    """ query edge by node """
    query_edge_by_node('09005700121709091548023739Y', df_topo)


    """ check for single direaction bfs """
    pid = '09005700121709031455036592S'
    rids = bfs(pid, df_topo, True)

    pid = '09005700121709091656461569Y'
    visited = set()
    bfs(pid, df_topo, True, visited, False)
    bfs(pid, df_topo_prev, False, visited, False)


    """ check for bidirection bfs """
    # 创科路北行
    # bidirection_bfs('09005700121709091038208679Y', df_topo, df_topo_prev)
    # 创科路南行
    # bidirection_bfs('09005700121709091041425059Y', df_topo, df_topo_prev)
    # 打石一路东行
    # bidirection_bfs('09005700121709091548023739Y', df_topo, df_topo_prev)
    # 打石一路西行
    bidirection_bfs('09005700121709091542295739Y', df_topo, df_topo_prev)
    # 特殊案例
    bidirection_bfs('09005700121709091542314649Y', df_topo, df_topo_prev, set(), True)

    
    """ plot node and its conncetions """
    plot_node_connections('09005700121709131151514569Y')


    """" combine rids """
    traj_rid_lst, rid_uf, rid_2_start = combine_rids(gdf_base, gdf_roads, plot=True)


    """ 根据rid 查询轨迹 """
    # 留仙洞
    get_trajectory_by_rid("988acb-1732-52e3-a58a-36eec3", rid_uf, gdf_roads)
    # fitian 
    df = get_trajectory_by_rid('550a27-40c5-f0d3-5717-a1907d', rid_uf, gdf_roads)
    from utils.geo_helper import gdf_to_geojson
    gdf_to_geojson(df, '../cache/panos_for_test')

#%%

traj_rid_lst, rid_uf, rid_2_start = combine_rids(gdf_base, gdf_roads, plot=True)

# get_trajectory_by_rid('586fa0-4623-8530-bcea-12e41d', rid_uf, traj_rid_lst, gdf_roads, plot=True)

get_trajectory_by_rid('52fec5-aae7-3da8-56ee-dfca06', rid_uf, traj_rid_lst, gdf_roads, plot=True)


# %%
