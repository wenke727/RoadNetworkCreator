#%%
import math
from matplotlib.pyplot import legend
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from collections import deque

from utils.pickle_helper import PickleSaver
from utils.geo_plot_helper import map_visualize
from utils.log_helper import LogHelper, logbook
from pano_base import pano_dict_to_gdf, extract_gdf_road_from_key_pano

logger = LogHelper(log_dir="../log", log_name='panos.log').make_logger(level=logbook.INFO)

#%%

def query_edge_by_node(id, df_topo):
    return df_topo.query( "src == @id or dst == @id" )


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
    # TODO sort
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
    graph = {}

    for src, dst, link in df_topo.reset_index()[['src', 'dst', 'link']].values:
        graph[src] = graph.get(src, set())
        graph[src].add(dst)

    return graph


def bfs(node, df_topo, plot=True, similar_threds=.9):
    if node not in df_topo.index:
        return []
    
    queue = deque([node])
    visited_pid = set()
    res = []
    
    while queue:
        cur_pid = queue.popleft()
        if cur_pid in visited_pid:
            continue
        
        # TODO: change to a more faster data structure
        for nxt_pid, nxt in df_topo.loc[cur_pid].iterrows():
            if nxt['similarity'] < similar_threds:
                continue
            if nxt_pid in visited_pid:
                continue
            if nxt_pid not in df_topo.index:
                continue
            
            queue.append(nxt_pid)
            if nxt['compressed_rid'] is not np.nan:
                res.append(nxt['compressed_rid'])
            if not nxt['link']:
                res.append(nxt['rid'])
            
            break

        visited_pid.add(cur_pid)
    
    if plot:
        map_visualize( gdf_roads.loc[res] )
    
    return res


def bidirection_bfs(pid, df_topo, df_topo_prev, visited_filter=True, plot=True):
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
    
    if pid not in df_topo.index or pid not in df_topo_prev.index:
        logger.warning(f"{pid} not in df_topo")
        return []

    rids_0 = bfs(pid, df_topo_prev, False)
    rids_1 = bfs(pid, df_topo, False)
    rids = rids_0[::-1] + rids_1

    if visited_filter:
        df_topo.query("rid not in @rids", inplace=True)
        df_topo_prev.query("rid not in @rids", inplace=True)
    
    if rids and plot:
        roads = gdf_roads.loc[rids]
        roads.loc[:, 'order'] = range(roads.shape[0])
        fig, ax = map_visualize(roads, color='gray')
        roads.plot(column='order', legend=True, ax=ax, linewidth=3)
    
    return rids


def combine_rids(plot=True):
    """Combining rids based on the amuzith similarity.

    Args:
        plot (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    df_topo, df_topo_prev = get_topo_from_gdf_pano(gdf_panos)

    queue = deque(gdf_roads.index.tolist())
    visited = set()
    res = {}

    while queue:
        cur = queue.popleft()
        if cur in visited:
            logger.debug(f"skip {cur}")
            continue
        
        cur_pid = gdf_roads.loc[cur]['src']
        rids = bidirection_bfs(cur_pid, df_topo, df_topo_prev, plot=False)
        
        if not rids:
            continue

        for rid in rids:
            visited.add(rid)
        res[cur] = rids
        logger.info(f"visited {cur_pid}/{cur}, links: {rids}, topo size {df_topo.shape[0]}")

    if plot:
        fig, ax = map_visualize(gdf_roads, scale=.05)
        gdf_roads.loc[list(visited)].plot(ax=ax)

    return res, visited


#%%
if __name__ == '__main__':

    pickler   = PickleSaver()
    pano_dict = pickler.read('../cache/pano_dict_lxd.pkl')
    gdf_panos = pano_dict_to_gdf(pano_dict)
    gdf_roads = extract_gdf_road_from_key_pano(gdf_panos)

    df_topo, df_topo_prev = get_topo_from_gdf_pano(gdf_panos)
    # graph = build_graph(df_topo)
    # topo = df_topo.to_dict(orient='index')


    """check for single pid"""
    pid = '09005700121709091037594139Y'
    res = _parse_road_and_links(gdf_panos.loc[pid])


    """ query edge by node """
    query_edge_by_node('09005700121709091548023739Y', df_topo)


    """ check for single direaction bfs """
    pid = '09005700121709031455036592S'
    rids = bfs(pid, df_topo, True)


    """ check for bidirection bfs """
    # it works at most time
    # 创科路北行
    # bidirection_bfs('09005700121709091038208679Y', df_topo, df_topo_prev)
    # 创科路南行
    # bidirection_bfs('09005700121709091041425059Y', df_topo, df_topo_prev)
    # 打石一路东行
    # bidirection_bfs('09005700121709091548023739Y', df_topo, df_topo_prev)
    # 打石一路西行
    bidirection_bfs('09005700121709091542295739Y', df_topo, df_topo_prev)

    # bidirection_bfs('09005700121709091041425059Y', df_topo, df_topo_prev)
    # bidirection_bfs('09005700122003241038503865O', df_topo, df_topo_prev)
    # bidirection_bfs('09005700121709091557348029Y', df_topo, df_topo_prev)


    """ plot node and its conncetions """
    plot_node_connections('09005700122003211407319405O')
    plot_node_connections('09005700122003211407335965O')


    """" combine rids """
    res, visited = combine_rids(plot=True)

#%%

# check unvistied rids
# node 1
plot_node_connections('09005700121709131151514569Y')

#%%
# node 2
# pid = '09005700121709091656421879Y'

pid = '09005700121709091657453129Y'
pid = '09005700121709091656461569Y'
# plot_node_connections(pid)
bidirection_bfs(pid, df_topo, df_topo_prev, False)



# %%
pid = '09005700121709091656461569Y'
plot_node_connections(pid)
# %%
