from typing import List

from tmll.tsp_legacy.models.response.tree.node import NodeTreeResponse


class TableTreeResponse:
    """A class to represent a table tree response from the TSP server.

    Attributes:
        nodes (List[NodeTreeResponse]): The list of nodes in the table tree.
    """

    def __init__(self, nodes: List[NodeTreeResponse]):
        self.nodes = nodes

    def __repr__(self) -> str:
        return f"TableTreeResponse(nodes={self.nodes})"
