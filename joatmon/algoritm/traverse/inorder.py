from collections import deque


def inorder(root):
    stack = deque()
    curr = root

    while stack or curr:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            yield curr.data
            curr = curr.right


def inorder_r(root):
    if root is None:
        return

    inorder_r(root.left)
    yield root.data
    inorder_r(root.right)
