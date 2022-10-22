from dataclasses import dataclass
from typing import List, Union

from .column import Column
from .column_group import ColumnGroup


@dataclass
class Group:
    members: List[Union[Column, ColumnGroup]]
