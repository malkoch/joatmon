from collections import deque


def postorder(root):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
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
        yield out.pop()


def postorder_r(root):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    if root is None:
        return

    postorder_r(root.left)
    postorder_r(root.right)
    yield root.data
