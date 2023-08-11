import pytest


def test_dfs():
    from joatmon.algoritm.search.dfs import dfs
    from joatmon.structure.graph import Graph

    g = Graph([(0, 1), (1, 2)], 3)

    ret = list(dfs(g, 0, [False, False, False]))
    assert ret == [0, 1, 2]


def test_dfs_r():
    from joatmon.algoritm.search.dfs import dfs_r
    from joatmon.structure.graph import Graph

    g = Graph([(0, 1), (1, 2)], 3)

    ret = list(dfs_r(g, 0, [False, False, False]))
    assert ret == [0]


if __name__ == '__main__':
    pytest.main([__file__])
