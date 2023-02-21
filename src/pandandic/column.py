from dataclasses import dataclass
from typing import Optional, Union

from .column_alias_not_yet_defined_exception import ColumnAliasNotYetDefinedException
from .defined_later import DefinedLater
from .type_definition import TypeDefinition


@dataclass
class Column(TypeDefinition):
    alias: Optional[Union[str, DefinedLater, DefinedLater.__class__]] = None

    def __set_name__(self, _, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is not None and hasattr(obj, "__getitem__"):
            return obj[self.column_name]
        else:
            return self

    @property
    def column_name(self) -> str:
        if self.alias == DefinedLater or isinstance(self.alias, DefinedLater):
            raise ColumnAliasNotYetDefinedException(self.name)
        return self.alias or self.name
