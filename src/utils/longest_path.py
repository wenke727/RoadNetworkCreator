 
from collections import defaultdict, deque

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
        

