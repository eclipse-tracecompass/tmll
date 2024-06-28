class NodeTree:
    """A class to represent a node in a tree from the TSP server.

    Attributes:
        name (str): The name of the node.
        id (int): The ID of the node.
        parent_id (int): The parent ID of the node.
    """

    def __init__(self, name: str, id: int, parent_id: int):
        self.name = name
        self.id = id
        self.parent_id = parent_id

    def __repr__(self) -> str:
        return f"NodeTree(name={self.name}, id={self.id}, parent_id={self.parent_id})"

    @classmethod
    def from_tsp_node(cls, tsp_node) -> "NodeTree":
        """Create a NodeTree object from a TSP node.

        Args:
            tsp_node (dict): The TSP node.

        Returns:
            NodeTree: The NodeTree object.
        """

        return cls(' '.join(tsp_node.labels), tsp_node.id, tsp_node.parent_id)
