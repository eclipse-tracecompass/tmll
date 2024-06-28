class Output:
    """A class to represent an output from the TSP server.

    Attributes:
        name (str): Output name.
        id (str): Output id.
        description (str): Output description.
        type (str): Output type.
        start (int): Output start time.
        end (int): Output end time.
        final (int): Output final time.
    """

    def __init__(self, name: str, id: str, description: str, type: str, start: int, end: int, final: int) -> None:
        self.name = name
        self.id = id
        self.description = description
        self.type = type
        self.start = start
        self.end = end
        self.final = final

    @classmethod
    def from_tsp_output(cls, tsp_output) -> 'Output':
        return cls(name=tsp_output.name, id=tsp_output.id, description=tsp_output.description, type=tsp_output.type, start=tsp_output.start, end=tsp_output.end, final=tsp_output.final)

    def __repr__(self) -> str:
        return f"Output(name={self.name}, id={self.id}, description={self.description}, type={self.type}, start={self.start}, end={self.end}, final={self.final})"
