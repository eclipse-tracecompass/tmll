from typing import List, Union

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

        :param tsp_tree: The TSP tree.
        :type tsp_tree: tsp.models.Tree
        :return: The Tree object.
        :rtype: Tree
        """
        nodes = tsp_tree.entries
        nodes = [NodeTree.from_tsp_node(node) for node in nodes]
        return cls(nodes)
    
    def get_node_by_id(self, node_id: int) -> Union[NodeTree, None]:
        """Get a node by its ID.

        :param node_id: The ID of the node.
        :type node_id: int
        :return: The node with the given ID.
        :rtype: Union[NodeTree, None]
        """
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_node_parent(self, node_id: int) -> Union[NodeTree, None]:
        """Get the parent of a node.

        :param node_id: The ID of the node.
        :type node_id: int
        :return: The parent of the node.
        :rtype: Union[NodeTree, None]
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return None
        return self.get_node_by_id(node.parent_id)
