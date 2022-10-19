from copy import deepcopy
from datetime import datetime, date
from numbers import Number
from typing import Dict, Tuple, Any

import pandas as pd
from pandas import DataFrame
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .column import Column


class BaseFrame:
    _typed_columns: Dict[str, Column] = None

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

    def read_csv(self, *args, **kwargs) -> DataFrame:
        if self.enforce_typed_columns:
            typed_columns = list(self.get_typed_columns().keys())
            if self.allowed_extra_columns:
                columns = self.read_csv_columns(*args, **kwargs)
                typed_columns = list(set(columns).union(typed_columns))
            kwargs["usecols"] = typed_columns
        self._apply_validation(kwargs)
        df = pd.read_csv(*args, **kwargs)
        return df

    def read_excel(self, *args, **kwargs) -> DataFrame:
        if self.enforce_typed_columns:
            typed_columns = list(self.get_typed_columns().keys())
            if self.allowed_extra_columns:
                columns = self.read_excel_columns(*args, **kwargs)
                typed_columns = list(set(columns).union(typed_columns))
            kwargs["usecols"] = typed_columns
        self._apply_validation(kwargs)
        df = pd.read_excel(*args, **kwargs)
        return df

    def read_parquet(self, *args, **kwargs) -> DataFrame:
        if self.enforce_typed_columns:
            typed_columns = list(self.get_typed_columns().keys())
            if self.allowed_extra_columns:
                columns = self.read_parquet_columns(*args, **kwargs)
                typed_columns = list(set(columns).union(typed_columns))
            kwargs["columns"] = typed_columns
        df = pd.read_parquet(*args, **kwargs)
        for key, column in filter(self._type_is_castable, self.get_typed_columns().items()):
            df.loc[:, key] = df.loc[:, key].astype(column.type)
        return df

    def _apply_validation(self, kwargs):
        if self.validate:
            kwargs["dtype"] = dict(
                map(lambda key_column_tuple: (key_column_tuple[0], key_column_tuple[1].type),
                    filter(self._type_is_not_date,
                           filter(self._type_is_not_any,
                                  self.get_typed_columns().items()))))
            kwargs["parse_dates"] = list(
                map(lambda key_column_tuple: key_column_tuple[0],
                    filter(self._type_is_date, self.get_typed_columns().items())))

    @staticmethod
    def _type_is_castable(key_column_tuple: Tuple[str, Column]) -> bool:
        return BaseFrame._type_is_not_any(key_column_tuple) and BaseFrame._type_is_not_date(key_column_tuple)

    @staticmethod
    def _type_is_not_any(key_column_tuple: Tuple[str, Column]) -> bool:
        _, column = key_column_tuple
        return column.type is not Any

    @staticmethod
    def _type_is_date(key_column_tuple: Tuple[str, Column]) -> bool:
        _, column = key_column_tuple
        return column.type in (datetime, "datetime", date, "date")

    @staticmethod
    def _type_is_not_date(key_column_tuple: Tuple[str, Column]) -> bool:
        return not BaseFrame._type_is_date(key_column_tuple)

    @staticmethod
    def read_csv_columns(*args, **kwargs) -> list:
        header_kwargs = deepcopy(kwargs)
        header_kwargs["nrows"] = 0
        if "usecols" in header_kwargs:
            del header_kwargs["usecols"]
        columns = pd.read_csv(*args, **header_kwargs).columns
        try:
            args[0].seek(0)
        except AttributeError:
            pass
        return columns.tolist()

    @staticmethod
    def read_excel_columns(*args, **kwargs) -> list:
        header_kwargs = deepcopy(kwargs)
        header_kwargs["nrows"] = 0
        if "usecols" in header_kwargs:
            del header_kwargs["usecols"]
        columns = pd.read_excel(*args, **header_kwargs).columns
        try:
            args[0].seek(0)
        except AttributeError:
            pass
        return columns.tolist()

    @staticmethod
    def read_parquet_columns(*args, **kwargs) -> list:
        from pyarrow.parquet import ParquetFile
        pq_file = ParquetFile(args[0])
        columns = pq_file.metadata.schema.names
        return columns

    @classmethod
    def get_typed_columns(cls) -> Dict[str, Column]:
        if cls._typed_columns is None:
            cls._typed_columns = dict(filter(lambda kv: isinstance(kv[1], Column), cls.__dict__.items()))
        return cls._typed_columns
