from typing import List

from tmll.common.models.timegraph.row import TimeGraphRow


class TimeGraph:
    """A class to represent a time graph from the TSP server.

    Attributes:
        rows (List[TimeGraphRow]): The rows of the time graph.
    """

    def __init__(self, rows: List[TimeGraphRow]) -> None:
        self.rows = rows

    def __repr__(self) -> str:
        return f"TimeGraph(rows={self.rows})"
    
    @classmethod
    def from_tsp_time_graph(cls, tsp_time_graph):
        rows = [TimeGraphRow.from_tsp_row(row) for row in tsp_time_graph.rows]
        return cls(rows)