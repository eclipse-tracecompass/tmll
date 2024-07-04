from typing import List


class TableDataRow:
    """The row of the table.

    Attributes:
        index (int): The index of the row.
        values (List[str]): The values of the row
    """

    def __init__(self, index: int, values: List[str]) -> None:
        self.index = index
        self.values = values

    def __repr__(self) -> str:
        return f"Row(index={self.index}, values={self.values})"

    @classmethod
    def from_tsp_row(cls, tsp_row):
        values = [cell.content for cell in tsp_row.cells]
        return cls(tsp_row.index, values)
