from dataclasses import dataclass
from typing import Type


@dataclass
class Column:
    type: Type

    def __set_name__(self, _, name):
        self.name = name
