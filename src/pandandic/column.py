from dataclasses import dataclass
from typing import Type, Any, Optional, Union

from .defined_later import DefinedLater


@dataclass
class Column:
    type: Type = Any
    alias: Optional[Union[str, DefinedLater, DefinedLater.__class__]] = None

    def __set_name__(self, _, name):
        self.name = name

    @property
    def column_name(self) -> Union[str, DefinedLater, DefinedLater.__class__]:
        return self.alias or self.name
