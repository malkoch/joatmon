import pytest


def test_postorder():
    from joatmon.algoritm.traverse.postorder import postorder
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(postorder(root)) == [1, 2, 3]


def test_postorder_r():
    from joatmon.algoritm.traverse.postorder import postorder_r
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(postorder_r(root)) == [3]


if __name__ == '__main__':
    pytest.main([__file__])
