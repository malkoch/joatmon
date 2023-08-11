import collections

import pytest


def test_bfs():
    from joatmon.algoritm.search.bfs import bfs
    from joatmon.structure.graph import Graph

    g = Graph([(0, 1), (1, 2)], 3)

    ret = list(bfs(g, 0, [False, False, False]))
    assert ret == [0, 1, 2]


def test_bfs_r():
    from joatmon.algoritm.search.bfs import bfs_r
    from joatmon.structure.graph import Graph

    g = Graph([(0, 1), (1, 2)], 3)
    q = collections.deque()
    q.append(0)

    ret = list(bfs_r(g, q, [False, False, False]))
    assert ret == [0, 1, 0, 2]


if __name__ == '__main__':
    pytest.main([__file__])
