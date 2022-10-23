from typing import Union

from .column_set import ColumnSet
from .column import Column


class ColumnGroupException(Exception):
    def __init__(self, column: str, first_column_group: Union[ColumnSet, Column], second_column_group: ColumnSet):
        self._column = column
        self._first_column_group = first_column_group
        self._second_column_group = second_column_group
        super().__init__()

    def __str__(self) -> str:
        return f"Error. Column {self._column} is matched by members in {self._first_column_group} and {self._second_column_group}."
