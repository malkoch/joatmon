import pytest


def test_inorder():
    from joatmon.algoritm.traverse.inorder import inorder
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(inorder(root)) == [1, 3, 2]


def test_inorder_r():
    from joatmon.algoritm.traverse.inorder import inorder_r
    from joatmon.structure.tree import Tree

    root = Tree(3)
    root.left = Tree(1)
    root.right = Tree(2)

    assert list(inorder_r(root)) == [3]


if __name__ == '__main__':
    pytest.main([__file__])
