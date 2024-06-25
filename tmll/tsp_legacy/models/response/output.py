class OutputResponse:
    """A class to represent an output response from the TSP server.

    Attributes:
        name (str): Output name.
        id (str): Output id.
    """

    def __init__(self, name: str, id: str) -> None:
        self.name = name
        self.id = id

    def __repr__(self) -> str:
        return f"OutputResponse(name={self.name}, id={self.id})"
