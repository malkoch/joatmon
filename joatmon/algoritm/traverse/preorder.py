from collections import deque


def preorder(root):
    if root is None:
        return

    stack = deque()
    stack.append(root)

    while stack:
        curr = stack.pop()

        yield curr.data
        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)


def preorder_r(root):
    if root is None:
        return

    yield root.data
    preorder_r(root.left)
    preorder_r(root.right)
