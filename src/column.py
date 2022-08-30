from dataclasses import dataclass
from typing import Type


@dataclass
class Column:
    type: Type
    alias: str | None = None
