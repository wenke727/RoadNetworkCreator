import geopandas as gpd
import pandas as pd
import coordTransform_py.CoordTransform_utils as ct
import matplotlib.pyplot as plt
from haversine import haversine
import sys
from collections import deque
from shapely.geometry import Point, LineString
import numpy as np

from utils.coord.coord_transfer import bd_coord_to_mc, bd_mc_to_coord, bd_mc_to_wgs
from utils.geo_plot_helper import map_visualize
from utils.classes import Node

"""
几乎没有什么可用的函数了，遍历的方式也已更正
针对的是百度地图
"""

def reverse_shp(geometry):
    """
    Reverse the polyline (geometry)

    Args:
        geometry (LineString): [description]

    Returns:
        [LineString]: [description]
    """
    return LineString(np.vstack(geometry.xy).T[::-1])


def check_degree(id, graph):
    if isinstance(id, Node):
        return id.val, id.indegree, id.outdegree, [x.val for x in id.prev], [x.val for x in id.nxt]

    return id, graph[id].indegree, graph[id].outdegree, [x.val for x in graph[id].prev], [x.val for x in graph[id].nxt]


def traverse_road(origin, graph):
    """
    Traverse the road, connect the segement accoding to spatial relationship
    """
    # TODO 光侨路一边的坐标体系；
    origin = graph[origin] if type(origin) == str else origin

    def _bfs(queue, visited, res):
        while queue:
            node = queue.popleft()
            neighbors = list(node.nxt)

            if node.outdegree == 1:
                if neighbors[0] in visited:
                    continue
                queue.append(neighbors[0])
                visited.add(neighbors[0])
                res.append((node.val, neighbors[0].val))
            else:
                print("travrese warning: ", node.val, ": ",
                      node.outdegree, "outdegree != 1")
                for nxt_node in neighbors:
                    print(f'\t 邻接点 {nxt_node.val}, ({nxt_node.indegree}, {nxt_node.outdegree}), ',  [
                          x.val for x in nxt_node.nxt])

                    if nxt_node not in visited and nxt_node.indegree == 1:
                        res.append((node.val, nxt_node.val))
                        queue.append(nxt_node)
                        visited.add(nxt_node)

    print(f"\torigin {origin.val}")
    queue = deque([origin])
    visited = set()
    res = []

    _bfs(queue, visited, res)

    return res


def query_road_by_OD(df_roads, lst):
    """Query roads by origin and destination

    Args:
        df_roads (GeoDataFrame): with `start` `end` `geometry`
        lst (List): [start, end...]

    Returns:
        [GeoDataFrame]: the query result
    """
    # segments = pd.DataFrame( [ (lst[i], lst[i+1]) for i in range( len(lst) - 1 ) ], columns=['start', 'end'] )
    segments = gpd.GeoDataFrame(lst, columns=['start', 'end'])
    return segments.merge(df_roads, on=['start', 'end'])


def identify_reverse_road(prev_node, cur_node, df_roads):
    """百度API部分道路的线形刚好相反，需要纠正

    Args:
        prev_node ([type]): [description]
        cur_node ([type]): [description]
        df_roads ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not cur_node.check_0_out_more_2_in():
        return cur_node
    
    from utils.spatialAnalysis import angle_bet_two_line
    print("identify_reverse_road: ", cur_node.val)

    # 更新错误方向的线段, 重新合并一下
    nxt_node = None
    base_line = {'x0': prev_node.x, 'y0': prev_node.y,
                 'x1': cur_node.x, 'y1': cur_node.y, 'node': prev_node}

    miminum = sys.maxsize
    for node in cur_node.prev:
        if node == prev_node:
            continue
        line = {'x0': cur_node.x, 'y0': cur_node.y,
                'x1': node.x, 'y1': node.y, 'node': node}
        angel = angle_bet_two_line(base_line, line)

        if miminum > min(abs(angel), abs(angel-180)):
            nxt_node = node
            miminum = min(abs(angel), abs(angel-180))

    cur_node.move_prev_to_nxt(nxt_node)

    # update the direction of wrong link in df_roads record
    record_index = df_roads.query(f"end == '{cur_node.val}' ").query( f"start =='{nxt_node.val}'").index
    df_roads.loc[record_index, 'start'], df_roads.loc[record_index,'end'] = df_roads.loc[record_index, 'end'], df_roads.loc[record_index, 'start']
    line = df_roads.loc[record_index, 'geometry']
    print(type(line))
    df_roads.loc[record_index, 'geometry'] = df_roads.loc[record_index].apply(lambda x: reverse_shp(x.geometry), axis=1)
    # 后续与之相连的也需要变更
    if nxt_node is not None:
        nxt_node.move_nxt_to_prev(cur_node)

    identify_reverse_road(cur_node, nxt_node, df_roads)


def traverse_road_consider_reverse_edge(origin, graph, df_roads):
    """traverse road consider reverse edge, based on the baidu searching API respond

    Args:
        origin ([type]): [description]
        graph ([type]): [description]
        df_roads ([type]): [description]

    Returns:
        [type]: [description]
    """
    res = traverse_road(origin, graph)
    count = 1
    # ! 判断写少了一个(), 函数没有执行
    while len(res) > 1 and graph[res[-1][1]].check_0_out_more_2_in():
        print(
            f"\n--------\n{res[-1][0]},{res[-1][1]}, couting {count}, {graph[res[-1][1]].check_0_out_more_2_in()}\n")

        identify_reverse_road(graph[res[-1][0]], graph[res[-1][1]], df_roads)

        res = traverse_road(origin, graph)
        count += 1
        if count >= 4:
            break
    return res, query_road_by_OD(df_roads, res)


def extract_roads_info(df_roads):
    """
    @DESC: Extract topo information (point, graph, zero degree point) from roads file
    """
    points = list(df_roads.start.values) + list(df_roads.end.values)
    indegree = {p: 0 for p in points}
    outdegree = {p: 0 for p in points}
    graph = {p: Node(p) for p in points}

    for index, item in df_roads.iterrows():
        graph[item.start].add(graph[item.end])
        indegree[item.end] += 1
        outdegree[item.start] += 1

    # # just for validation
    # for key in outdegree:
    #     a = graph[key].outdegree - outdegree[key]
    #     b = graph[key].indegree - indegree[key]
    #     if a != 0 or b!= 0:
    #         print(key)

    zero_indegree_points = [x for x in points if indegree[x] == 0]

    points = pd.concat([pd.DataFrame(indegree,  index=['indegree']).T,
                        pd.DataFrame(outdegree, index=['outdegree']).T], axis=1)
    points = gpd.GeoDataFrame( points.reset_index().rename(columns={'index': 'coords'}))
    points.geometry = points.apply(lambda x: Point(
            ct.bd09_to_wgs84(*bd_mc_to_coord(*[float(i) for i in x.coords.split(',')]))
        ), axis=1)
    
    points.set_index('coords', inplace=True)
    # points.to_file('光侨路_点.geojson', driver='GeoJSON')

    return points, graph, zero_indegree_points


def calculate_adj_points_dis(df_origin, inplace=False):
    df = df_origin if inplace else df_origin.copy()
    df.loc[:, 'x1'], df.loc[:, 'y1'] = df.x.shift(1), df.y.shift(1)
    return df.apply(lambda i: haversine((i.y, i.x), (i.y1, i.x1))*1000, axis=1)


def create_crawl_point(road_one_way: gpd.GeoDataFrame, geometry_type='point', visualize=False):
    """Extract all the points of the road and arrange them according to the spatial position

    Args:
        road_one_way (gpd.GeoDataFrame): Road with one-way or the one side
        visualize (bool, optional): [description]. Defaults to False.

    Returns:
        [gpd.GeoDataFrame]: Road points in order
    """

    df_order_coords = []
    coords_lst = road_one_way.geometry.apply(lambda x: np.vstack(x.xy).T)
    for index, i in enumerate(coords_lst):
        for j in i:
            df_order_coords.append(j)

    df_order_coords = pd.DataFrame(df_order_coords, columns=['x', 'y'])

    df_order_coords.loc[:, 'dis'] = calculate_adj_points_dis(
        df_order_coords, inplace=True)
    df_order_coords.query('dis > 0 ', inplace=True)
    df_order_coords['dis_cum'] = df_order_coords.dis.rolling(
        df_order_coords.shape[0], min_periods=1).sum()

    if geometry_type == 'point':
        geometry = df_order_coords.apply(lambda i: Point(i.x, i.y), axis=1)
    else:
        geometry = df_order_coords.apply(
            lambda x: LineString([(x.x, x.y), (x.x1, x.y1)]), axis=1)
    df_order_coords = gpd.GeoDataFrame(df_order_coords, geometry=geometry)

    df_order_coords.reset_index(drop=True, inplace=True)
    df_order_coords.loc[:, 'id'] = df_order_coords.index
    # TODO coords values: start or end port
    df_order_coords.loc[:, 'coords'] = df_order_coords.apply(
        lambda x:  bd_coord_to_mc(*ct.wgs84_to_bd09(x.x1, x.y1)), axis=1)

    if visualize:
        map_visualize(df_order_coords)

    # df_order_coords.to_file('./光侨路_南行_节点.geojson', driver="GeoJSON")
    return df_order_coords


if __name__ == '__main__':

    df_roads = gpd.read_file(
        '/home/pcl/traffic/RoadNetworkCreator_by_View/input/光侨路.geojson')
    df_roads.query("start=='12685799.40,2586251.61' ")

    points, graph, zero_indegree_points = extract_roads_info(df_roads)
    res, df = traverse_road_consider_reverse_edge(
        '12685054.46,2591594.75', graph, df_roads)
    df_roads.query("end=='12685799.40,2586251.61' ")

    map_visualize(df, 's')
    df
    pass




