from typing import List

from tmll.tsp_legacy.models.response.trace import TraceResponse


class ExperimentResponse(TraceResponse):
    """A class to represent an experiment response from the TSP server.

    Attributes:
        name (str): The name of the experiment.
        uuid (str): The UUID of the experiment.
        start (int): The start time of the experiment.
        end (int): The end time of the experiment.
        num_events (int): The number of events in the experiment.
        indexing (str): The indexing of the experiment.
        traces (List[TraceResponse]): The list of traces in the experiment.
    """

    def __init__(self, name: str, uuid: str, start: int, end: int, num_events: int, indexing: str, traces: List[TraceResponse] = []) -> None:
        super().__init__(name, uuid, start, end, num_events, indexing)
        self.traces = traces

    @classmethod
    def from_trace_response(cls, trace_response: TraceResponse, traces: List[TraceResponse] = []) -> 'ExperimentResponse':
        return cls(trace_response.name, trace_response.uuid, trace_response.start, trace_response.end, trace_response.num_events, trace_response.indexing, traces)

    def __repr__(self) -> str:
        return f"ExperimentResponse(name={self.name}, uuid={self.uuid}, start={self.start}, end={self.end}, num_events={self.num_events}, indexing={self.indexing}, traces={self.traces})"
