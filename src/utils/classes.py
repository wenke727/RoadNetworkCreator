import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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
    def __init__(self, v=0, edges=None, *args, **kwargs):

        self.Vertex = v
        self.Edge = 0

        # key is node, value is neighbors
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

    def build_graph(self, edges):
        for edge in edges:
            self.add_edge(*edge)
        return self.graph

    def calculate_degree(self,):
        df_degree = pd.merge(
            pd.DataFrame([[key, len(self.prev[key])]
                          for key in self.prev], columns=['coord', 'indegree']),
            pd.DataFrame([[key, len(self.graph[key])]
                          for key in self.graph], columns=['coord', 'outdegree']),
            on='coord'
        )

        # df_degree = gpd.GeoDataFrame(df_degree,
        #                              geometry=df_degree.coord.apply(
        #                                  lambda x: Point(*x))
        #                              )

        self.degree = df_degree

        # df_degree.query( " indegree > 1 or outdegree >1 " )

