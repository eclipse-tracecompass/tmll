class Trace:
    """A class to represent a trace from the TSP server.

    Attributes:
        name (str): The name of the trace.
        uuid (str): The UUID of the trace.
        start (int): The start time of the trace.
        end (int): The end time of the trace.
        num_events (int): The number of events in the trace.
        indexing (str): The indexing of the trace.
    """

    def __init__(self, name: str, uuid: str, path: str, start: int, end: int, num_events: int, indexing: str) -> None:
        self.name = name
        self.uuid = uuid
        self.path = path
        self.start = start
        self.end = end
        self.num_events = num_events
        self.indexing = indexing

    def __repr__(self) -> str:
        return f"Trace(name={self.name}, uuid={self.uuid}, start={self.start}, end={self.end}, num_events={self.num_events}, indexing={self.indexing})"

    @classmethod
    def from_tsp_trace(cls, tsp_trace) -> 'Trace':
        return cls(name=tsp_trace.name, uuid=tsp_trace.UUID, path=tsp_trace.path, start=tsp_trace.start, end=tsp_trace.end, num_events=tsp_trace.number_of_events, indexing=tsp_trace.indexin_status)
