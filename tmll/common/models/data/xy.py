from typing import List


class XYData:
    """A class to represent an XY data from the TSP server.

    Attributes:
        x_values (List[int]): The list of x values in the XY data.
        y_values (List[float]): The list of y values in the XY data.
    """

    def __init__(self, x_values: List[int], y_values: List[float]) -> None:
        self.x_values = x_values
        self.y_values = y_values

    def __repr__(self) -> str:
        return f"XYData(x_values={self.x_values}, y_values={self.y_values})"
