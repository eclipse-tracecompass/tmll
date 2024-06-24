from typing import List


class TableDataRowResponse:
    """The row response of the table.

    Attributes:
        index (int): The index of the row.
        values (List[str]): The values of the row
    """

    def __init__(self, index: int, values: List[str]) -> None:
        self.index = index
        self.values = values

    def __repr__(self) -> str:
        return f"Row(index={self.index}, values={self.values})"