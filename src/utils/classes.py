import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from collections import defaultdict, deque

from .geo_plot_helper import map_visualize


class Node:
    """
    Define the node in the road network 
    """

    def __init__(self, id):
        self.val = id
        self.x, self.y = [float(i) for i in id.split(',')]
        self.prev = set()
        self.nxt = set()
        self.indegree = 0
        self.outdegree = 0

    def add(self, point):
        self.nxt.add(point)
        self.outdegree += 1

        point.prev.add(self)
        point.indegree += 1

    def check_0_out_more_2_in(self):
        return self.outdegree == 0 and self.indegree >= 2

    def move_nxt_to_prev(self, node):
        if node not in self.nxt:
            return False

        self.nxt.remove(node)
        self.prev.add(node)
        self.indegree += 1
        self.outdegree -= 1
        return True

    def move_prev_to_nxt(self, node):
        if node not in self.prev:
            return False

        self.prev.remove(node)
        self.nxt.add(node)
        self.indegree -= 1
        self.outdegree += 1
        return True

class Digraph:
    def __init__(self, edges=None, *args, **kwargs):
        """[summary]

        Args:
            edges ([list], optional): [description]. Defaults to None.
        """
        self.graph = {}
        self.prev = {}
        if edges is not None:
            self.build_graph(edges)

        self.calculate_degree()

    def __str__(self):
        return ""

    def add_edge(self, start, end):
        for p in [start, end]:
            for g in [self.graph, self.prev]:
                if p in g:
                    continue
                g[p] = set()

        self.graph[start].add(end)
        self.prev[end].add(start)
        pass

    def remove_edge(self, start, end):
        self.graph[start].remove(end)
        if len(self.graph[start]) == 0:
            del self.graph[start]
        
        self.prev[end].remove(start)
        if len(self.prev[end]) == 0:
            del self.prev[end]
        pass

    def build_graph(self, edges):
        for edge in edges:
            self.add_edge(*edge)
        return self.graph

    def clean_empty_set(self):
        for item in [self.prev, self.graph]:
            for i in list(item.keys()):
                if len(item[i]) == 0:
                    del item[i]
        pass
        
    def calculate_degree(self,):
        self.clean_empty_set()
        self.degree = pd.merge(
            pd.DataFrame([[key, len(self.prev[key])]
                          for key in self.prev], columns=['node_id', 'indegree']),
            pd.DataFrame([[key, len(self.graph[key])]
                          for key in self.graph], columns=['node_id', 'outdegree']),
            how='outer',
            on='node_id'
        ).fillna(0).astype(np.int)
        
        return self.degree

    def get_origin_point(self,):
        self.calculate_degree()
        return self.degree.query( "indegree == 0 and outdegree != 0" ).node_id.values
    
    def _combine_edges_helper(self, origins, result=None, pre=None, roads=None, vis=False):
        """combine segment based on the node degree

        Args:
            origins ([type]): [description]
            result (list, optional): [Collection results]. Defaults to None.
            pre ([type], optional): The previous points, the case a node with more than 2 children. Defaults to None.
            roads (gpd.Geodataframe, optional): 道路数据框，含有属性 's' 和 'e'. Defaults to None.
            vis (bool, optional): [description]. Defaults to False.
        """
        for o in origins:
            pre_node = o
            path = []
            if pre is not None:
                path = [[pre,o]]
                self.remove_edge(pre,o)

            # case: 0 indegree, > 2 outdegree
            if len(self.graph[o]) > 1:
                o_lst = list( self.graph[o] )
                self._combine_edges_helper( o_lst, result, o, roads, vis )
                return
            
            while o in self.graph and len(self.graph[o]) == 1:
                o = list(self.graph[o])[0]
                self.remove_edge( pre_node, o )
                path.append([pre_node, o])
                pre_node = o

            if roads is not None:
                assert hasattr(roads, 's') and hasattr(roads, 'e'), "attribute is missing"
                tmp = gpd.GeoDataFrame(path, columns=['s','e']).merge( roads, on=['s','e'] )
            
                ids = []
                for i in tmp.rid.values:
                    if len(ids) == 0 or ids[-1] != i:
                        ids.append(i)
                # ids = '_'.join(map(str, ids))

                if vis: map_visualize(tmp, 's')
                if result is not None: result.append([tmp, ids ])

            else:
                if result is not None: result.append([path, []])
            
        return

    def combine_edges(self, roads=None, vis=False):
        import copy
        graph_bak = copy.deepcopy(self.graph)
        prev_back = copy.deepcopy(self.prev.copy())
        
        result = [] # path, road_id
        origins = self.get_origin_point()
        while len(origins) > 0:
            self._combine_edges_helper(origins, result, roads=roads)
            origins = self.get_origin_point()

        if roads is not None and vis:
            for i, _ in result:
                map_visualize(i, 's')
        
        self.graph = graph_bak
        self.prev = prev_back
        
        return result


class LongestPath:
    """
    @param n: The number of nodes
    @param starts: One point of the edge
    @param ends: Another point of the edge
    @param lens: The length of the edge
    @return: Return the length of longest path on the tree.
    """
    def __init__(self, edges:pd.DataFrame, origin):
        starts, ends, lens = edges.start.values, edges.end.values, edges.length.values
        graph = self.build_graph(starts, ends, lens)
        self.graph = graph
        
        start, _, _  = self.bfs_helper(graph, origin)
        end, self.length, path = self.bfs_helper(graph, start)
        self.path = self.get_path(start, end, path)

        return
    
    def build_graph(self, starts, ends, lens):
        graph = defaultdict(list)
        for i in range(len(starts)):
            graph[starts[i]].append((starts[i], ends[i], lens[i]))
            graph[ends[i]].append((ends[i], starts[i], lens[i]))
            
        return graph

    def bfs_helper(self, graph, start):
        queue = deque([(start, 0)])
        path = {start: None}
        end, max_length = 0, 0
        
        while queue:
            cur, sum_length = queue.pop()
            max_length = max(max_length, sum_length)
            if max_length == sum_length:
                end = cur

            for _, nxt, edge_len in graph[cur]:
                if nxt in path:
                    continue

                path[nxt] = cur
                queue.appendleft((nxt, sum_length + edge_len))

        return end, max_length, path

    def get_path(self, start, end, visit):
        res = []
        cur = end
        while cur in visit:
            res.append(cur)
            cur = visit[cur]
        
        return res[::-1]

