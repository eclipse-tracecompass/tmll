from typing import Optional, TypeVar, Generic

T = TypeVar('T')


class Response(Generic[T]):
    """A class to represent a response from the TSP server.

    Attributes:
        error (str): The error message of the response.
        result (T): The result of the response.
    """

    def __init__(self, error: Optional[str] = None, result: Optional[T] = None):
        self.error = error
        self.result = result

    def __repr__(self) -> str:
        return f"Response(error={self.error}, result={self.result})"
