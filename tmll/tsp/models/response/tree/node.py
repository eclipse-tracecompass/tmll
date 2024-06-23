class NodeTreeResponse:
    """A class to represent a node in a tree response from the TSP server.

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
        return f"NodeTreeResponse(name={self.name}, id={self.id}, parent_id={self.parent_id})"
