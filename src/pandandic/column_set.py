from dataclasses import dataclass, field
from typing import Type, List, Any, Union

from .column_set_members_not_yet_defined_exception import ColumnSetMembersNotYetDefinedException
from .defined_later import DefinedLater


@dataclass
class ColumnSet:
    members: Union[List[str], DefinedLater, DefinedLater.__class__]  # list of columns matched to this set
    type: Type = Any  # dtype applied to this set
    regex: bool = False  # enables matching members by regex
    _consumed_columns: List[str] = field(default_factory=list)

    def __set_name__(self, _, name):
        self.name = name

    def _consume(self, column: str) -> None:
        self._consumed_columns.append(column)

    def reset(self) -> None:
        self._consumed_columns.clear()

    @property
    def columns(self) -> List[str]:
        if self.members == DefinedLater or isinstance(self.members, DefinedLater):
            raise ColumnSetMembersNotYetDefinedException(self.name)

        if self.regex:
            return self._consumed_columns
        else:
            return list(filter(self._consumed_columns.__contains__, self.members))
