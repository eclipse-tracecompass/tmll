class TableDataColumnResponse:

    def __init__(self, id: str, name: str, description: str, type: str):
        self.id = id
        self.name = name
        self.description = description
        self.type = type

    def __repr__(self) -> str:
        return f"TableColumnResponse(id={self.id}, name={self.name}, description={self.description}, type={self.type})"
