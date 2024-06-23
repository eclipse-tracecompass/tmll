class TraceResponse:
    """A class to represent a trace response from the TSP server.

    Attributes:
        name (str): The name of the trace.
        uuid (str): The UUID of the trace.
        start (int): The start time of the trace.
        end (int): The end time of the trace.
        num_events (int): The number of events in the trace.
        indexing (str): The indexing of the trace.
    """

    def __init__(self, name: str, uuid: str, start: int, end: int, num_events: int, indexing: str) -> None:
        self.name = name
        self.uuid = uuid
        self.start = start
        self.end = end
        self.num_events = num_events
        self.indexing = indexing

    def __repr__(self) -> str:
        return f"TraceResponse(name={self.name}, uuid={self.uuid}, start={self.start}, end={self.end}, num_events={self.num_events}, indexing={self.indexing})"
