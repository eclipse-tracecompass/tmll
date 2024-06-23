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


class TableDataResponse:
    """The response for a table data type.

    Attributes:
        columns (List[str]): The columns of the table.
        rows (List[Row]): The rows of the table.
    """

    def __init__(self, columns: List[str], rows: List[TableDataRowResponse]) -> None:
        self.columns = columns
        self.rows = rows

    def __repr__(self) -> str:
        return f"TableResponse(columns={self.columns}, rows={self.rows})"
