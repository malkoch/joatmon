class Graph:
    def __init__(self, edges, n):
        self.adj_list = [[] for _ in range(n)]

        for (src, dest) in edges:
            self.adj_list[src].append(dest)
            self.adj_list[dest].append(src)
