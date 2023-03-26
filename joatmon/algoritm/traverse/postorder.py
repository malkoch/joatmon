from collections import deque


def postorder(root):
    if root is None:
        return

    stack = deque()
    stack.append(root)
    out = deque()

    while stack:
        curr = stack.pop()
        out.append(curr.data)
        if curr.left:
            stack.append(curr.left)
        if curr.right:
            stack.append(curr.right)

    while out:
        print(out.pop(), end=' ')


def postorder_r(root):
    if root is None:
        return

    postorder_r(root.left)
    postorder_r(root.right)
    print(root.data, end=' ')
