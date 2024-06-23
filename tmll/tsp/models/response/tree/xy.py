from typing import List

from tmll.tsp.models.response.tree.table import TableTreeResponse
from tmll.tsp.models.response.tree.node import NodeTreeResponse


class XYTreeResponse(TableTreeResponse):
    """A class to represent an XY tree response from the TSP server.

    Attributes:
        nodes (List[NodeTreeResponse]): The list of nodes in the XY tree.
    """

    def __init__(self, nodes: List[NodeTreeResponse]):
        super().__init__(nodes)

    def __repr__(self) -> str:
        return f"XYTreeResponse(nodes={self.nodes})"

    @classmethod
    def from_table_tree_response(cls, table_tree_response: TableTreeResponse) -> 'XYTreeResponse':
        return cls(table_tree_response.nodes)
