from dataclasses import dataclass
from typing import Type, Any, Optional


@dataclass
class TypeDefinition:
    type: Type = Any
    datetime_options: Optional[dict] = None  #: Passed as kwargs to pd.to_datetime.
    numeric_options: Optional[dict] = None  #: Passed as kwargs to pd.to_numeric.
    drop_na: bool = False
