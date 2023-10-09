from joatmon.structure.tree import (
    Node,
    Tree
)


class MyData:
    def __init__(self, name, amount=None):
        self.name = name
        self.amount = amount

    def __str__(self):
        if self.amount is not None:
            return f'{self.name=:} {self.amount=:}'
        else:
            return f'{self.name=:}'

    def __repr__(self):
        return str(self)


tree = Tree()
level_2_item = Node(MyData('iron_head'))

level_1_item_1 = Node(MyData('stone_head', 1))
level_1_item_2 = Node(MyData('iron', 50))
level_1_item_3 = Node(MyData('string', 100))

level_0_item_1_1 = Node(MyData('stick', 1))
level_0_item_1_2 = Node(MyData('stone', 5))
level_0_item_1_3 = Node(MyData('string', 10))

level__1_item_1_1_1 = Node(MyData('wood', 5))

level_0_item_1_1.add_node(level__1_item_1_1_1)

level_1_item_1.add_node(level_0_item_1_1)
level_1_item_1.add_node(level_0_item_1_2)
level_1_item_1.add_node(level_0_item_1_3)

level_2_item.add_node(level_1_item_1)
level_2_item.add_node(level_1_item_2)
level_2_item.add_node(level_1_item_3)

tree.root = level_2_item
tree.visualize()
