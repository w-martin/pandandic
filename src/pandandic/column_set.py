from dataclasses import dataclass
from typing import Type, List, Any, Union

from .defined_later import DefinedLater


@dataclass
class ColumnSet:
    members: Union[List[str], DefinedLater, DefinedLater.__class__]  # list of columns matched to this set
    type: Type = Any  # dtype applied to this set
    regex: bool = False  # enables matching members by regex

    def __set_name__(self, _, name):
        self.name = name
