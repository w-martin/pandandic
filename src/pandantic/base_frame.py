from copy import deepcopy
from typing import Dict

import pandas as pd
from pandas import DataFrame
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .column import Column


class BaseFrame:

    def __init__(self):
        self.validate = True
        self.enforce_typed_columns = True
        self.allowed_extra_columns = False

    def with_validation(self, validate: bool = True) -> Self:
        self.validate = validate
        return self

    def with_enforce_typed_columns(self, enforce_typed_columns: bool = True) -> Self:
        self.enforce_typed_columns = enforce_typed_columns
        return self

    def with_allowed_extra_columns(self, allowed_extra_columns: bool = True) -> Self:
        self.allowed_extra_columns = allowed_extra_columns
        return self

    def from_csv(self, *args, **kwargs) -> DataFrame:
        if self.enforce_typed_columns:
            typed_columns = list(self.get_typed_columns().keys())
            if self.allowed_extra_columns:
                columns = self.peek_columns_from_csv(*args, **kwargs)
                typed_columns = list(set(columns).union(typed_columns))
            kwargs["usecols"] = typed_columns
        df = pd.read_csv(*args, **kwargs)
        return df

    @staticmethod
    def peek_columns_from_csv(*args, **kwargs) -> pd.Index:
        header_kwargs = deepcopy(kwargs)
        header_kwargs["nrows"] = 0
        if "usecols" in header_kwargs:
            del header_kwargs["usecols"]
        columns = pd.read_csv(*args, **header_kwargs).columns
        try:
            args[0].seek(0)
        except AttributeError:
            pass
        return columns

    @classmethod
    def get_typed_columns(cls) -> Dict[str, Column]:
        return dict(filter(lambda kv: isinstance(kv[1], Column), cls.__dict__.items()))
