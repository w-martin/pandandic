class ColumnAliasNotYetDefinedException(Exception):
    def __init__(self, column_name: str):
        self._column_name = column_name

    def __str__(self) -> str:
        return f"Error. Column with name {self._column_name} was used with an alias that is not defined."
