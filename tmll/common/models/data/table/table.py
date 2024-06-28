from typing import List

from tmll.common.models.data.table.row import TableDataRow


class TableData:
    """The table data.

    Attributes:
        columns (List[str]): The columns of the table.
        rows (List[Row]): The rows of the table.
    """

    def __init__(self, columns: List[str], rows: List[TableDataRow]) -> None:
        self.columns = columns
        self.rows = rows

    def __repr__(self) -> str:
        return f"Table(columns={self.columns}, rows={self.rows})"
