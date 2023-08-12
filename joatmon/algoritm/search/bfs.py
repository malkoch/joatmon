from collections import deque


def bfs(graph, v, discovered):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    q = deque()
    discovered[v] = True
    q.append(v)

    while q:
        v = q.popleft()
        yield v

        for u in graph.adj_list[v]:
            if not discovered[u]:
                discovered[u] = True
                q.append(u)


def bfs_r(graph, q, discovered):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if not q:
        return

    v = q.popleft()
    yield v

    for u in graph.adj_list[v]:
        if not discovered[u]:
            discovered[u] = True
            q.append(u)

    for x in bfs_r(graph, q, discovered):
        yield x
