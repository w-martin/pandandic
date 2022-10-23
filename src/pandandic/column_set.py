from dataclasses import dataclass
from typing import Type, List


@dataclass
class ColumnSet:
    type: Type  # dtype applied to this set
    members: List[str]  # list of columns matched to this set
    regex: bool = False  # enables matching members by regex

    def __set_name__(self, _, name):
        self.name = name
