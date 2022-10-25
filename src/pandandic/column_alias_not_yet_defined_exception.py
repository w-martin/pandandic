from .column import Column


class ColumnAliasNotYetDefinedException(Exception):
    def __init__(self, column: Column):
        self._column = column

    def __str__(self) -> str:
        return f"Error. Column with name {self._column.name} was used with an alias that is not defined."
