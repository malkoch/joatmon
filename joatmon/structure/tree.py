class Node:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, data):
        self.data = data
        self.branches = []

    def add_node(self, node):
        self.branches.append(node)

    def visualize(self, depth):
        print(self.data)
        for idx, item in enumerate(self):
            print('|', end='')
            for d in range(depth):
                print('  |', end='')

            print('__' if idx == 0 or idx == len(self) - 1 else '__', end='')

            item.visualize(depth + 1)

    def __len__(self):
        return len(self.branches)

    def __iter__(self):
        for item in self.branches:
            yield item


class Tree:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
        self.root = None

    def visualize(self):
        if self.root is None:
            return

        self.root.visualize(0)
