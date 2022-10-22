from dataclasses import dataclass, field
from typing import Type, List


@dataclass
class ColumnGroup:
    type: Type
    members: List[str]
    regex: bool = False  # enables matching members by regex

    def __set_name__(self, _, name):
        self.name = name
