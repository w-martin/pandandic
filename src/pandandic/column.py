from dataclasses import dataclass
from typing import Type, Any, Optional, Union

from .column_alias_not_yet_defined_exception import ColumnAliasNotYetDefinedException
from .defined_later import DefinedLater


@dataclass
class Column:
    type: Type = Any
    alias: Optional[Union[str, DefinedLater, DefinedLater.__class__]] = None

    def __set_name__(self, _, name):
        self.name = name

    @property
    def column_name(self) -> str:
        if self.alias == DefinedLater or isinstance(self.alias, DefinedLater):
            raise ColumnAliasNotYetDefinedException(self.name)
        return self.alias or self.name
