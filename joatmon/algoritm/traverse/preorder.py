from collections import deque


def preorder(root):
    if root is None:
        return

    stack = deque()
    stack.append(root)

    while stack:
        curr = stack.pop()

        print(curr.data, end=' ')
        if curr.right:
            stack.append(curr.right)
        if curr.left:
            stack.append(curr.left)


def preorder_r(root):
    if root is None:
        return

    print(root.data, end=' ')
    preorder_r(root.left)
    preorder_r(root.right)
