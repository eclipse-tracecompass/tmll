from typing import Optional


class TimeGraphState:
    """A class to represent a state in a time graph.

    Attributes:
        start (int): The start time of the state.
        end (int): The end time of the state.
        label (Optional[str]): The label of the state.
    """

    def __init__(self, start: int, end: int, label: Optional[str] = None) -> None:
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self) -> str:
        return f"TimeGraphState(start={self.start}, end={self.end}, label={self.label})"
    
    @classmethod
    def from_tsp_state(cls, tsp_state):
        return cls(tsp_state.start_time, tsp_state.end_time, tsp_state.label if hasattr(tsp_state, 'label') else None)