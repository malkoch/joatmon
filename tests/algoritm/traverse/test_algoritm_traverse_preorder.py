import pytest


def test_preorder():
    from joatmon.algoritm.traverse.preorder import preorder
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(preorder(root)) == [3, 1, 2]


def test_preorder_r():
    from joatmon.algoritm.traverse.preorder import preorder_r
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(preorder_r(root)) == [3]


if __name__ == '__main__':
    pytest.main([__file__])
