class TableDataColumn:

    def __init__(self, id: str, name: str, description: str, type: str):
        self.id = id
        self.name = name
        self.description = description
        self.type = type

    def __repr__(self) -> str:
        return f"TableColumn(id={self.id}, name={self.name}, description={self.description}, type={self.type})"

    @classmethod
    def from_tsp_table_column(cls, column) -> "TableDataColumn":
        return cls(column.id, column.name, column.description, column.type)