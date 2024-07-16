from typing import List

from tmll.common.models.trace import Trace


class Experiment(Trace):
    """A class to represent an experiment from the TSP server.

    Attributes:
        name (str): The name of the experiment.
        uuid (str): The UUID of the experiment.
        start (int): The start time of the experiment.
        end (int): The end time of the experiment.
        num_events (int): The number of events in the experiment.
        indexing (str): The indexing of the experiment.
        traces (List[Trace]): The list of traces in the experiment.
    """

    def __init__(self, name: str, uuid: str, start: int, end: int, num_events: int, indexing: str, traces: List[Trace] = []) -> None:
        super().__init__(name=name, uuid=uuid, start=start, end=end, num_events=num_events, indexing=indexing, path="")
        self.traces = traces

    @classmethod
    def from_trace(cls, trace_: Trace, traces: List[Trace] = []) -> 'Experiment':
        return cls(trace_.name, trace_.uuid, trace_.start, trace_.end, trace_.num_events, trace_.indexing, traces)

    @classmethod
    def from_tsp_experiment(cls, tsp_experiment) -> 'Experiment':
        traces = [Trace.from_tsp_trace(tsp_trace) for tsp_trace in tsp_experiment.traces.traces]
        return cls(tsp_experiment.name, tsp_experiment.UUID, tsp_experiment.start, tsp_experiment.end, tsp_experiment.number_of_events, tsp_experiment.indexing_status.name, traces)

    def __repr__(self) -> str:
        return f"Experiment(name={self.name}, uuid={self.uuid}, start={self.start}, end={self.end}, num_events={self.num_events}, indexing={self.indexing}, traces={self.traces})"
