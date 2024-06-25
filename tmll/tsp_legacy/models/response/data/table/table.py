from typing import List

from tmll.tsp_legacy.models.response.data.table.row import TableDataRowResponse


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
