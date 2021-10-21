#%%
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from collections import deque
from shapely.geometry import LineString

from utils.unionFind import UnionFind
from utils.pickle_helper import PickleSaver
from utils.azimuth_helper import azimuthAngle
from utils.log_helper import LogHelper, logbook
from utils.geo_plot_helper import map_visualize
from utils.geo_helper import coords_pair_dist, gdf_to_geojson, gdf_to_postgis
from pano_base import pano_dict_to_gdf, extract_gdf_roads_from_key_pano, extract_gdf_panos_from_key_pano, extract_gdf_roads


#%%
class Pano_UnionFind(UnionFind):
    def __init__(self, lst, gdf_roads=None, gdf_panos=None):
        super().__init__(lst)
        self.trajs = {}
        self.unvisited = set(lst) # 存在单点的情况
        self.gdf_roads = gdf_roads
        self.gdf_panos = gdf_panos

    def get_panos(self, rid, plot=False):
        rids = self.get_traj(rid, False)
        if rids is None:
            return None
        
        lst = []
        for sub in self.gdf_roads.loc[rids].Panos.apply(lambda x: [i['PID'] for i in x]).tolist():
            lst += sub
        
        if plot and self.gdf_panos is not None:
            map_visualize(self.gdf_panos.loc[lst])
        
        return self.gdf_panos.loc[lst].reset_index()

    def get_traj(self, rid=None, pid=None, plot=False):
        assert rid is not None or pid is not None, 'check input'
        if rid is None:
            rid = self.gdf_panos.loc[pid].RID
        
        root = self.find(rid)
        if root not in self.trajs:
            return None

        if plot and self.gdf_roads is not None:
            roads = self.gdf_roads.loc[self.trajs[root]]
            roads.loc[:, 'order'] = range(roads.shape[0])
            fig, ax = map_visualize(roads, color='black')
            roads.plot(column='order', legend=True, ax=ax, linewidth=3)
        
        return self.trajs[root]

    def set_traj(self, rid, traj):
        if rid in self.trajs:
            print(f"{rid} existed")
            return False
        
        self.trajs[rid] = traj
        
        return True

    def connect_traj(self, rids):
        self.unvisited.remove(rids[0])
        for i in range(len(rids)-1):
            self.unvisited.remove(rids[i+1])
            self.connect(rids[i], rids[i+1])
        
        self.set_traj(rids[0], rids)

    def trajs_to_gdf(self,):
        lst = {}
        for key in tqdm(self.trajs.keys()):
            lst[key] = {}
            lst[key]['rids'] = self.get_traj(rid=key, plot=False)
            lst[key]['rids_num'] = len(lst[key]['rids'])
            lst[key]['pids_df'] =  self.get_panos(key, plot=False) 
            lst[key]['pids_num'] = lst[key]['pids_df'].shape[0]
            # if lst[key]['pids_df'] is not None:
                # lst[key]['pred'] = pred_trajectory(lst[key]['pids_df'], df_pred_memo, aerial_view=False, combine_view=True, with_lanes=True)

        df = gpd.GeoDataFrame(lst).T
        df.rids = df.rids.astype(str)
        df.drop_duplicates('rids', inplace=True)
        df.rids = df.rids.apply(lambda x: eval(x))

        df.loc[:, 'geometry'] = df.apply(
            lambda x: 
                LineString(self.get_panos(x.name, False).geometry.apply(lambda x: x.coords[0]).tolist()),
            axis = 1
        )
        
        self.gdf = df.sort_values('pids_num', ascending=False)

        return self.gdf


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


def plot_node_connections(node, df_topo, *args, **kwargs):
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


def get_topo_from_gdf_pano(gdf_base, neg_dir_rids=set(), drop_irr_records=True, std_deviation=20):
    """Extract topo infomation from panos geodataframe.

    Args:
        gdf_base (geodataframe): [description]
        drop_irr_records (bool, optional): Remove irrelevant records (the origin or destiantion not in the key_panos). Defaults to True.

    Returns:
        [geodataframe]: [description]
    """

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
        if _rid in neg_dir_rids:
            res = [{'rid': _rid, 'src': dst, 'dst': src, 'link':False}]
        else:
            res = [{'rid': _rid, 'src': src, 'dst': dst, 'link':False}]

        for link in _links:
            info = {'src': pano.name, 'dst': link['PID'], 'link':True}
            res.append(info)
            
        return res

    def _cal_observ_prob(x):
        observ_prob_factor = 1 / (np.sqrt(2*np.pi) * std_deviation)
        return observ_prob_factor * np.exp(-np.power(x, 2)/(2*np.power(std_deviation, 2)))

    def _norm_observ_prob(df):
        df.loc[:, 'dist_prob_std'] = np.sqrt(df.dist_prob / df.dist_prob.max())
        df.loc[~df.link, 'dist_prob_std'] = 1
        
        return df

    def _cal_turn_sim(item, forward_dir=True):
        src, dst = item.src, item.dst
        a, b = gdf_base.loc[src].geometry.coords[0]
        c, d = gdf_base.loc[dst].geometry.coords[0]

        turn_azimuth = azimuthAngle(a, b, c, d)
        if forward_dir:
            turn_smi = (math.cos(azimuth_diff(item.dir_0, turn_azimuth))+1)/2
        else:
            turn_smi = (math.cos(azimuth_diff(item.dir_1, turn_azimuth))+1)/2
        
        return turn_smi

    def _post_process(df, att):
        df = df.groupby(att).apply(_norm_observ_prob)
        df.loc[:, 'turn_smi'] = df.apply(lambda x: _cal_turn_sim(x, True if att=='src' else False), axis=1)
        df.loc[:, 'similarity'] = df.apply(lambda x: x.cos_sim * x.cos_sim * x.dist_prob_std * x.turn_smi, axis=1)
        df.loc[~df.link, 'similarity'] = 1
        df.sort_values([att, 'link', 'similarity'], ascending=[True, True, False], inplace=True)
        
        return df

    topo = []
    for lst in tqdm(gdf_base.apply(lambda x: _parse_road_and_links(x), axis=1).values, "Parse road and link: "):
        topo += lst
    df_topo = pd.DataFrame(topo)
    df_topo.drop_duplicates(df_topo.columns, inplace=True)

    # drop irrelevant record
    if drop_irr_records:
        pano_ids = gdf_base.index.values.tolist()
        df_topo.query("src in @pano_ids and dst in @pano_ids", inplace=True)

    # compressed single rid
    compressed_pids = df_topo.query('dst==src')[['src', 'rid']].set_index('src').to_dict('index')
    df_topo.query('dst != src', inplace=True)
    df_topo.loc[:, 'compressed_rid'] = df_topo.apply(lambda x: compressed_pids[x.src]['rid'] if x.src in compressed_pids else np.nan, axis=1)
    
    # calculate the similarity 
    df_topo.loc[:,['dir_0', 'dir_1']] = df_topo.apply(
            lambda x: {'dir_0': gdf_base.loc[x.src]['MoveDir'], 'dir_1': gdf_base.loc[x.dst]['MoveDir']}, 
            axis=1, 
            result_type='expand'
    )
    df_topo.loc[:, 'cos_sim'] = df_topo.apply(lambda x: (math.cos(azimuth_diff(x.dir_0, x.dir_1))+1)/2, axis=1)

    # add distance factor
    df_topo.loc[:, 'dist'] = df_topo.apply(lambda x: 
        coords_pair_dist(gdf_base.loc[x.src].geometry, gdf_base.loc[x.dst].geometry), axis=1)
    df_topo.loc[:,'dist_prob'] = df_topo.apply(lambda x: _cal_observ_prob(x.dist), axis=1)
    df_topo = _post_process(df_topo, 'src')

    df_topo_prev = df_topo.copy()
    df_topo.set_index(['src', 'dst'], inplace=True)

    # inverse the graph
    df_topo_prev = _post_process(df_topo_prev, 'dst')
    df_topo_prev.set_index(['dst', 'src'], inplace=True)

    return df_topo, df_topo_prev


def bfs(node, df_topo, direction=True, visited=set(), plot=False, similar_threds=.7, logger=None):
    if node not in df_topo.index:
        return []
    
    queue, res = deque([node]), []
    internal_visited = set() # Avoid going backwards
    while queue:
        cur_pid = queue.popleft()
        if cur_pid not in df_topo.index:
            continue

        if logger is not None:
            info = df_topo.loc[cur_pid][['link','cos_sim', 'dist_prob_std','similarity']].reset_index()
            logger.debug(f"{cur_pid}, {'forword' if direction else 'backward'}:\n {info}")
        
        for nxt_pid, nxt in df_topo.loc[cur_pid].iterrows():
            connector = (cur_pid, nxt_pid) if direction else (nxt_pid, cur_pid)
            connector_revert = connector[::-1]
            if logger is not None:
                logger.debug(f"\t\tlink: {connector}, visited: {connector in visited}")
            
            if nxt['similarity'] < similar_threds or connector in visited or connector in internal_visited:
                continue
            
            if nxt['compressed_rid'] is not np.nan and nxt['compressed_rid'] not in res:
                if (nxt['compressed_rid'], nxt['compressed_rid']) not in visited:
                    res.append(nxt['compressed_rid'])
                    visited.add((nxt['compressed_rid'], nxt['compressed_rid']))
            if not nxt['link']:
                res.append(nxt['rid'])
            
            queue.append(nxt_pid)
            visited.add(connector)
            internal_visited.add(connector_revert)

            break
    
    if plot and len(res) > 0:
        map_visualize( gdf_roads.loc[res] )
    
    return res


def bidirection_bfs(pid, df_topo, df_topo_prev, visited=set(), similar_threds=.7, plot=False, logger=None):
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
    if pid not in df_topo.index and pid not in df_topo_prev.index:
        if logger is not None:
            logger.warning(f"{pid} not in df_topo")
        return []

    params = {'visited':visited, "plot":False, 'logger':logger, 'similar_threds':similar_threds}
    rids_0 = bfs(pid, df_topo_prev, False, **params)
    rids_1 = bfs(pid, df_topo, True, **params)
    rids = rids_0[::-1] + rids_1
   
    if rids and plot:
        roads = gdf_roads.loc[rids]
        roads.loc[:, 'order'] = range(roads.shape[0])
        fig, ax = map_visualize(roads, color='black')
        roads.plot(column='order', legend=True, ax=ax, linewidth=3)
    
    return rids


def combine_rids(gdf_base, gdf_roads, gdf_panos, plot=True, logger=None):
    """Combining rids based on the amuzith similarity.

    Args:
        plot (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    neg_dir_rids = set(gdf_panos[gdf_panos.Order<0].RID.unique())
    df_topo, df_topo_prev = get_topo_from_gdf_pano(gdf_base, neg_dir_rids)

    res            = {}
    rid_2_start    = {}
    uf             = Pano_UnionFind(gdf_roads.index.tolist(), gdf_roads, gdf_panos)
    queue          = deque(gdf_roads.query('src != dst').index.tolist())  # 起点和终点都是一样的路段不适宜作为遍历的起始点
    visited        = set() # {(src, dst)}
    dulplicate_src = []
    
    while queue:
        cur = gdf_roads.loc[queue.popleft()]
        cur_pid = cur['src']

        if (cur_pid, cur['dst']) in visited:
            if logger is not None:
                logger.debug(f"skip {cur.name}")
            continue
        
        rids = bidirection_bfs(cur_pid, df_topo, df_topo_prev, visited, 0.7, False, logger)
        if not rids:
            continue

        for i in rids:
            if i in rid_2_start:
                if logger is not None:
                    logger.warning(f"{i} exist in rid_2_start")
            rid_2_start[i] = rid_2_start.get(i, set())
            rid_2_start[i].add(rids[0])
        
        if rids[0] not in res:
            res[rids[0]] = [rids]
        else:
            res[rids[0]].append(rids)
            dulplicate_src.append(rids[0])
            print(f"dulplicate: {rids[0]}")

        uf.connect_traj(rids)

        if logger is not None:
            logger.debug(f"visited {cur_pid}/{cur.name}, links: {rids}")

    if plot:
        fig, ax = map_visualize(gdf_roads, scale=.05)
        gdf_roads.merge(pd.DataFrame(visited, columns=['src', 'dst']), on=['src', 'dst']).plot(ax=ax)

    assert len(dulplicate_src) == 0, "check the logic of combine_rids to process"

    # return uf, res, rid_2_start
    return uf, df_topo, df_topo_prev



#%%
if __name__ == '__main__':
    logger = LogHelper(log_dir="../log", log_name='pano_topo.log', stdOutFlag=False).make_logger(level=logbook.INFO)

    pickler   = PickleSaver()
    # pano_dict = pickler.read('../cache/pano_dict_lxd.pkl')
    pano_dict = pickler.read('../cache/pano_dict_futian.pkl')
    gdf_base = pano_dict_to_gdf(pano_dict)
    gdf_panos = extract_gdf_panos_from_key_pano(gdf_base, update_dir=True)
    gdf_roads = extract_gdf_roads(gdf_panos)


    """" combine rids """
    uf, df_topo, df_topo_prev = combine_rids(gdf_base, gdf_roads, gdf_panos, plot=False, logger=logger)
    df_trajs = uf.trajs_to_gdf()
    gdf_to_postgis(df_trajs, 'test_topo_futian_new')

    
    """ query edge by node """
    query_edge_by_node('09005700121709091548023739Y', df_topo)


    """ plot node and its conncetions """
    # 没有被访问到节点
    plot_node_connections('09005700122003221437254103O', df_topo)
    plot_node_connections('09005700122003271208195303O', df_topo)


    """ check for single direaction bfs """
    pid = '09005700121709031455036592S'
    rids = bfs(pid, df_topo, True)

    pid = '09005700121709091656461569Y'
    visited = set()
    bfs(pid, df_topo, True, visited, False)
    bfs(pid, df_topo_prev, False, visited, False)


    """ check for bidirection bfs """
    # bidirection_bfs('09005700121709091038208679Y', df_topo, df_topo_prev) # 创科路北行
    # bidirection_bfs('09005700121709091041425059Y', df_topo, df_topo_prev) # 创科路南行
    # bidirection_bfs('09005700121709091548023739Y', df_topo, df_topo_prev) # 打石一路东行
    bidirection_bfs('09005700121709091542295739Y', df_topo, df_topo_prev, logger=logger)# 打石一路西行
    bidirection_bfs('09005700122003271208195303O', df_topo, df_topo_prev, set(), .7, True, logger=logger) # 特殊案例


    """ 根据rid 查询轨迹 """
    # 留仙洞
    # get_trajectory_by_rid("988acb-1732-52e3-a58a-36eec3", rid_2_start, traj_rid_lst, gdf_roads)
    # futian 
    rid = '550a27-40c5-f0d3-5717-a1907d' # 金田路福田地铁站附近
    uf.get_traj(rid, plot=True)

    rid = 'edbf2d-e2f3-703f-4b9f-9d6819' # 深南大道-市民中心-东行掉头
    uf.get_traj(rid, plot=True)

    rid = 'd51f52-4ab6-cba6-dc4f-2fdf73' # 深南大道/益田路立交-东侧
    uf.get_traj(rid, plot=True)

    rid = '514cba-89ea-b8d6-3de2-15f9ac' # 深南大道/益田路立交-西侧
    uf.get_traj(rid, plot=True)

    rid = '113422-7515-f096-fb9f-ec2bce'
    uf.get_traj(rid, plot=True)

    bidirection_bfs('09005700122003221437282513O', df_topo, df_topo_prev, set(), .7, True, logger=logger) # 特殊案例
    
