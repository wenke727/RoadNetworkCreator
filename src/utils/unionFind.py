class UnionFind():
    def __init__(self, lst):
        self.father = {i: i for i in lst}
        
    def connect(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        
        if root_a != root_b:
            self.father[root_b] = root_a
    
    def find(self, x):
        if self.father[x] == x:
            return x
        
        self.father[x] = self.find(self.father[x])
        
        return self.father[x] 
