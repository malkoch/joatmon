from collections import deque


def dfs(graph, v, discovered):
    stack = deque()
    stack.append(v)

    while stack:
        v = stack.pop()
        if discovered[v]:
            continue

        discovered[v] = True
        yield v

        # do for every edge (v, u)
        adj_list = graph.adj_list[v]
        for i in reversed(range(len(adj_list))):
            u = adj_list[i]
            if not discovered[u]:
                stack.append(u)


def dfs_r(graph, v, discovered):
    discovered[v] = True  # mark the current node as discovered
    yield v

    for u in graph.adj_list[v]:
        if not discovered[u]:  # if `u` is not yet discovered
            dfs_r(graph, u, discovered)
