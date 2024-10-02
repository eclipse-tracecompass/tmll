from typing import List

from tmll.common.models.timegraph.state import TimeGraphState


class TimeGraphRow:
    """A class to represent a row in a time graph.

    Attributes:
        entry_id (int): The entry ID of the row.
        states (List[TimeGraphState]): The states of the row.
    """

    def __init__(self, entry_id: int, states: List[TimeGraphState]) -> None:
        self.entry_id = entry_id
        self.states = states

    def __repr__(self) -> str:
        return f"TimeGraphRow(entry_id={self.entry_id}, states={self.states})"
    
    @classmethod
    def from_tsp_row(cls, tsp_row):
        states = [TimeGraphState.from_tsp_state(state) for state in tsp_row.states]
        return cls(tsp_row.entry_id, states)