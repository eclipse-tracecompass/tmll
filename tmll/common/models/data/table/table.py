from typing import List

from tmll.common.models.data.table.row import TableDataRow


class TableData:
    """The table data.

    Attributes:
        columns (List[str]): The columns of the table.
        rows (List[Row]): The rows of the table.
    """

    def __init__(self, size: int, index: int, columns: List[str], rows: List[TableDataRow]) -> None:
        self.size = size
        self.index = index
        self.columns = columns
        self.rows = rows

    def __repr__(self) -> str:
        return f"Table(columns={self.columns}, rows={self.rows})"

    @classmethod
    def from_tsp_table(cls, tsp_table):
        rows = [TableDataRow.from_tsp_row(row) for row in tsp_table.lines]
        return cls(tsp_table.size, tsp_table.low_index, tsp_table.column_ids, rows)
