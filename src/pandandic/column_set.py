from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd

from .column_set_members_not_yet_defined_exception import ColumnSetMembersNotYetDefinedException
from .defined_later import DefinedLater
from .type_definition import TypeDefinition


@dataclass
class ColumnSet(TypeDefinition):
    """

    """

    #: List of columns matched to this set.
    members: Union[List[str], DefinedLater, DefinedLater.__class__] = DefinedLater
    regex: bool = False  #: Enables matching members by regex
    _consumed_columns: List[str] = field(default_factory=list)

    def __set_name__(self, _, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is not None and hasattr(obj, "__getitem__"):
            return obj[self.columns]
        else:
            return self

    def consume_column(self, column: str) -> None:
        """
        Consumes the given column.
        """
        self._consumed_columns.append(column)

    def reset(self) -> None:
        self._consumed_columns.clear()

    @property
    def columns(self) -> List[str]:
        if self.members == DefinedLater or isinstance(self.members, DefinedLater):
            raise ColumnSetMembersNotYetDefinedException(self.name)

        if self.regex:
            return self._consumed_columns
        elif len(self._consumed_columns):
            return list(filter(self._consumed_columns.__contains__, self.members))
        else:
            return self.members
