from typing import List

from tmll.common.models.tree.node import NodeTree


class Tree:
    """A class to represent a tree from the TSP server.

    Attributes:
        nodes (List[NodeTree]): The list of nodes in the tree.
    """

    def __init__(self, nodes: List[NodeTree]):
        self.nodes = nodes

    def __repr__(self) -> str:
        return f"TableTree(nodes={self.nodes})"
    
    @classmethod
    def from_tsp_tree(cls, tsp_tree) -> "Tree":
        """Create a Tree object from a TSP tree.

        Args:
            tsp_tree (dict): The TSP tree.

        Returns:
            Tree: The Tree object.
        """
        nodes = tsp_tree.entries
        nodes = [NodeTree.from_tsp_node(node) for node in nodes]
        return cls(nodes)
