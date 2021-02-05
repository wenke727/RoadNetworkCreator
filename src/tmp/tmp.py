import pickle
import pandas as pd
from pano_base import intersection_visulize
from db.db_process import extract_connectors_from_panos_respond
from db.db_process import load_from_DB
from road_matching import get_panos_of_road_by_id
from pano_base import intersection_visulize
from road_network import OSM_road_network
from utils.spatialAnalysis import linestring_length
from utils.geo_plot_helper import map_visualize

from utils.longest_path import Solution
from haversine import haversine
 
from collections import defaultdict, deque

DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(False)
linestring_length(DB_roads, True)


osm_shenzhen = pickle.load(open("/home/pcl/traffic/data/input/road_network_osm_shenzhen.pkl", 'rb') )
df_nodes, df_edges = osm_shenzhen.nodes, osm_shenzhen.edges
road = df_edges.query( f" rid == 529070115 " )


# ! 将匹配的`匝道`删除, 可能不是一棵树
matching = get_panos_of_road_by_id(362735582, df_edges, vis=True)

pid = '09005700011601080935054018N'
DB_connectors.query( f"prev_pano_id=='{pid}'" )

tmp =  intersection_visulize( pid, scale = 3 )


DB_pano_base.query( f"ID=='{pid}' " ).iloc[0].Links




connecters = extract_connectors_from_panos_respond( DB_pano_base, DB_roads )
connecters.query( f"prev_pano_id=='{pid}'" )

# 计算连接器的长度
links = connecters.query( f"prev_pano_id in {matching.PID_end.values.tolist()}" )


tmp = links.merge( DB_panos[['PID', 'geometry']], left_on='prev_pano_id', right_on='PID' )
links.loc[:, 'length'] = tmp.apply( lambda x:  haversine( x.geometry_x.coords[:][0], x.geometry_y.coords[:][0] ), axis=1 ) * 1000
links.query( f"prev_pano_id=='{pid}'" )



a = links[['prev_pano_id', 'PID', 'length']].rename( columns={"prev_pano_id": 'PID_start', 'PID':'PID_end'} )
a.loc[:, 'links'] = True
b = matching[['PID_start', 'PID_end', 'length']]
edges = pd.concat( [a, b] ).fillna(False)

 



class Solution:
    """
    @param n: The number of nodes
    @param starts: One point of the edge
    @param ends: Another point of the edge
    @param lens: The length of the edge
    @return: Return the length of longest path on the tree.
    """
    def longestPath(self, n, starts, ends, lens):
        graph = self.build_graph(starts, ends, lens)
        self.graph = graph
        
        # start, _, _  = self.bfs_helper(graph, starts[0])
        start, _, _  = self.bfs_helper(graph, starts[0])
        end, l, path = self.bfs_helper(graph, start)
        self.path = self.get_path(start, end, path)

        return l
    
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
        



a = Solution( )

a.longestPath( 100, edges.PID_start.values, edges.PID_end.values, edges.length.values )
a.graph[pid]


path = [ (a.path[i+1], a.path[i])  for i in range( len(a.path)-1)]
path = pd.DataFrame(path, columns=['PID_start', 'PID_end'])

path.merge( edges, on=['PID_start','PID_end'] )



map_visualize( DB_roads.merge( path, on=['PID_start','PID_end'] ) )

# ! 因为有些连接点的links是一进一出的
intersection_visulize( '09005700121709091539584169Y', scale = 5 )


edges.query( "PID_end == '09005700121709091539584169Y' " )


# %%
