from dataclasses import dataclass, field
from typing import List, Union

from .column import Column
from .column_set import ColumnSet


@dataclass
class ColumnGroup:
    members: List[Union[Column, ColumnSet]]

    def __set_name__(self, _, name):
        self.name = name
