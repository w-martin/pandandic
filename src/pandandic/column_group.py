from dataclasses import dataclass
from typing import List, Union

from more_itertools import flatten

from .column import Column
from .column_set import ColumnSet


@dataclass
class ColumnGroup:
    members: List[Union[Column, ColumnSet, "ColumnGroup"]]

    def __set_name__(self, _, name):
        self.name = name

    @property
    def columns(self) -> List[str]:
        return list(flatten(map(self._get_columns, self.members)))

    def _get_columns(self, member: Union[Column, ColumnSet, "ColumnGroup"]) -> List[str]:
        if isinstance(member, Column):
            return [member.column_name]
        else:
            return member.columns
