from dataclasses import dataclass
from typing import List, Union

from .column import Column
from .column_set import ColumnSet


@dataclass
class ColumnGroup:
    members: List[Union[Column, ColumnSet]]
