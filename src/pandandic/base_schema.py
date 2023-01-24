import re
import sys
from copy import deepcopy
from datetime import datetime, date
from numbers import Number
from typing import Dict, Tuple, Any, List, Union, Callable

import numpy as np
import pandas as pd
from more_itertools import consume

from .type_definition import TypeDefinition

try:
    import dask.dataframe as dd
except:
    ReturnType = pd.DataFrame
else:
    ReturnType = Union[pd.DataFrame, dd.DataFrame]

from .column import Column
from .column_alias_not_yet_defined_exception import ColumnAliasNotYetDefinedException
from .column_group import ColumnGroup
from .column_group_exception import ColumnGroupException
from .column_set import ColumnSet
from .column_set_members_not_yet_defined_exception import ColumnSetMembersNotYetDefinedException
from .defined_later import DefinedLater

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


# noinspection PyTypeChecker
class BaseSchema:
    """
    Enabled schema-in-code by subclassing BaseSchema.
    This enables wrapping of pandas and dask read functions, and provides other utility such as column references.
    See https://github.com/w-martin/pandandic for usage documentation or to submit bug reports.
    """
    _column_map: Dict[str, Column] = None
    _column_set_map: Dict[str, ColumnSet] = None
    _column_group_map: Dict[str, ColumnGroup] = None

    enforce_types = True
    enforce_columns = True
    allow_extra_columns = False
    greedy_column_sets = False
    dask = False

    def __init__(self):
        self._get_column_map()
        self._get_column_set_map()
        self._get_column_group_map()

    def with_enforced_types(self, enforce_types: bool = True) -> Self:
        self.enforce_types = enforce_types
        return self

    def with_enforced_columns(self, enforce_typed_columns: bool = True) -> Self:
        self.enforce_columns = enforce_typed_columns
        return self

    def with_extra_columns_allowed(self, allowed_extra_columns: bool = True) -> Self:
        self.allow_extra_columns = allowed_extra_columns
        return self

    def with_greedy_column_sets(self, greedy_column_sets: bool = True) -> Self:
        self.greedy_column_sets = greedy_column_sets
        return self

    def with_dask(self, dask: bool = True) -> Self:
        self.dask = dask
        return self

    def read_csv(self, *args, **kwargs) -> ReturnType:
        if self.enforce_columns or self.enforce_types:
            columns = self.read_csv_columns(*args, **kwargs)
            column_map = self._compute_column_map(columns)

            if self.enforce_columns:
                allowed_columns = list(map(self._first, column_map))
                if self.allow_extra_columns:
                    allowed_columns = list(set(allowed_columns).union(columns))

                kwargs["usecols"] = allowed_columns

            if self.enforce_types:
                self._add_validation_kwargs(column_map, kwargs)

            df = pd.read_csv(*args, **kwargs)

            if self.enforce_types:
                df = self._apply_column_spec(column_map, df)
        else:
            df = pd.read_csv(*args, **kwargs)
        return df

    @staticmethod
    def _first(pair: Tuple[Any, Any]) -> Any:
        return pair[0]

    def read_excel(self, *args, **kwargs) -> ReturnType:
        if self.enforce_columns or self.enforce_types:
            columns = self.read_excel_columns(*args, **kwargs)
            column_map = self._compute_column_map(columns)

            if self.enforce_columns:
                allowed_columns = list(map(self._first, column_map))
                if self.allow_extra_columns:
                    allowed_columns = list(set(allowed_columns).union(columns))

                kwargs["usecols"] = allowed_columns

            if self.enforce_types:
                self._add_validation_kwargs(column_map, kwargs)

            df = pd.read_excel(*args, **kwargs)

            if self.enforce_types:
                df = self._apply_column_spec(column_map, df)
        else:
            df = pd.read_excel(*args, **kwargs)
        return df

    def read_parquet(self, *args, **kwargs) -> ReturnType:
        columns = self.read_parquet_columns(*args, **kwargs)
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(map(self._first, column_map))
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            kwargs["columns"] = allowed_columns

        df = pd.read_parquet(*args, **kwargs)
        if self.enforce_types:
            df = self._apply_column_spec(column_map, df)
        return df

    def read_avro(self, *args, **kwargs) -> ReturnType:
        from pandavro import read_avro

        columns = self.read_avro_columns(args[0])
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(map(self._first, column_map))
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            kwargs["columns"] = allowed_columns

        df = read_avro(*args, **kwargs)
        if self.enforce_types:
            df = self._apply_column_spec(column_map, df)
        return df

    def apply(self, df) -> pd.DataFrame:
        df = df.copy()

        columns = df.columns.tolist()
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(map(self._first, column_map))
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            df = df[allowed_columns]

        if self.enforce_types:
            df = self._apply_column_spec(column_map, df)
        return df

    def _apply_column_spec(self, columns: List[Tuple[str, TypeDefinition]],
                           df: pd.DataFrame) -> pd.DataFrame:
        for column_name, type_definition in columns:

            if type_definition.drop_na:
                df = df.dropna(subset=[column_name])

            if self._type_is_numeric(type_definition.type):
                numeric_options = type_definition.numeric_options or {}
                df[column_name] = pd.to_numeric(df[column_name], **numeric_options)
                df[column_name] = df[column_name].astype(type_definition.type)

            elif self._type_is_date(type_definition.type):
                datetime_options = type_definition.datetime_options or {}
                df[column_name] = pd.to_datetime(df[column_name], **datetime_options)

            elif self._type_is_not_any(type_definition.type):
                df[column_name] = df[column_name].astype(type_definition.type)
        return df

    def get_columns(self, item: str) -> List[str]:
        if self._column_map is not None and item in self._column_map:
            return [self._column_map[item].column_name]

        if self._column_set_map is not None and item in self._column_set_map:
            return self._column_set_map[item].columns

        if self._column_group_map is not None and item in self._column_group_map:
            return self._column_group_map[item].columns
        return []

    def _compute_column_map(self, columns: List[str]) -> List[Tuple[str, TypeDefinition]]:
        consume(map(lambda cs: cs.reset(), self._column_set_map.values()))

        key_column_map = {(column.alias or column.name): column
                          for column in self._get_column_map().values()}
        for alias, column in key_column_map.items():
            if alias == DefinedLater or isinstance(alias, DefinedLater):
                raise ColumnAliasNotYetDefinedException(column)

        if len(self._get_column_set_map()) == 0:
            return list(map(tuple, key_column_map.items()))

        column_bag = np.array([key_column_map[c] if c in key_column_map else None for c in columns])
        consumed_columns = ~np.equal(column_bag, None)

        for key, column_set in self._get_column_set_map().items():
            if column_set.members == DefinedLater or isinstance(column_set.members, DefinedLater):
                raise ColumnSetMembersNotYetDefinedException(column_set)

        exact_column_sets = list(filter(lambda column_set: not column_set.regex, self._get_column_set_map().values()))
        regex_column_sets = list(filter(lambda column_set: column_set.regex, self._get_column_set_map().values()))

        for i, column in enumerate(columns):
            for column_set in exact_column_sets:
                if column in column_set.members:
                    if consumed_columns[i]:
                        if not self.greedy_column_sets:
                            raise ColumnGroupException(column, column_bag[i], column_set)
                    else:
                        consumed_columns[i] = True
                        column_bag[i] = column_set
                        self._column_set_map[column_set.name].consume_column(column)

            for column_set in regex_column_sets:
                if any((re.match(member, column) for member in column_set.members)):
                    if consumed_columns[i]:
                        if not self.greedy_column_sets:
                            raise ColumnGroupException(column, column_bag[i], column_set)
                    else:
                        consumed_columns[i] = True
                        column_bag[i] = column_set
                        self._column_set_map[column_set.name].consume_column(column)

        result = {columns[i]: column_or_group
                  for i, column_or_group in
                  filter(lambda ix: ix[1] is not None, enumerate(column_bag))}
        if self.enforce_columns:
            for column_set in exact_column_sets:
                for column in column_set.members:
                    if column not in result:
                        result[column] = column_set
        return list(map(tuple, result.items()))

    def _add_validation_kwargs(self, columns: List[Tuple[str, TypeDefinition]], kwargs: Dict[str, Any]):
        kwargs["dtype"] = dict(
            map(self._column_type_from_tuple,
                filter(self._filter_on_type(self._type_is_not_date),
                   filter(self._filter_on_type(self._type_is_not_any),
                          columns))))

        kwargs["parse_dates"] = list(
            map(self._column_from_tuple,
                filter(self._filter_on_type(self._type_is_date),
                       columns)))

    @staticmethod
    def _filter_on_type(f: Callable[[type], bool]) -> Callable[[Tuple[str, type]], bool]:
        def _apply_given_filter(column_type_tuple: Tuple[str, TypeDefinition]) -> bool:
            _, type_definition = column_type_tuple
            return f(type_definition.type)

        return _apply_given_filter

    @staticmethod
    def _column_from_tuple(column_type_tuple: Tuple[str, TypeDefinition]) -> str:
        column, _ = column_type_tuple
        return column

    @staticmethod
    def _column_type_from_tuple(column_type_tuple: Tuple[str, TypeDefinition]) -> Tuple[str, type]:
        column, type_definition = column_type_tuple
        return column, type_definition.type

    @staticmethod
    def _type_is_not_any(t: type) -> bool:
        return t not in (Any, None)

    @staticmethod
    def _type_is_date(t: type) -> bool:
        return t in (datetime, "datetime", date, "date")

    @classmethod
    def _type_is_not_date(cls, t: type) -> bool:
        return not cls._type_is_date(t)

    @staticmethod
    def _type_is_numeric(t: type) -> bool:
        return isinstance(t, Number) or t in (int, float, complex)

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
    def read_parquet_columns(*args, **__) -> list:
        from pyarrow.parquet import ParquetFile
        pq_file = ParquetFile(args[0])
        columns = pq_file.metadata.schema.names
        return columns

    @staticmethod
    def read_avro_columns(filepath_or_buffer) -> list:
        import fastavro
        if isinstance(filepath_or_buffer, str):
            with open(filepath_or_buffer, "rb") as f:
                reader = fastavro.reader(f)
                row = next(reader, dict())
        else:
            reader = fastavro.reader(filepath_or_buffer)
            row = next(reader, dict())
            filepath_or_buffer.seek(0)
        return list(row.keys())

    @classmethod
    def _get_column_map(cls) -> Dict[str, Column]:
        if cls._column_map is None:
            cls._column_map = dict(filter(lambda kv: isinstance(kv[1], Column), cls.__dict__.items()))
        return cls._column_map

    @classmethod
    def _get_column_set_map(cls) -> Dict[str, ColumnSet]:
        if cls._column_set_map is None:
            cls._column_set_map = dict(filter(lambda kv: isinstance(kv[1], ColumnSet), cls.__dict__.items()))
        return cls._column_set_map

    @classmethod
    def _get_column_group_map(cls) -> Dict[str, ColumnGroup]:
        if cls._column_group_map is None:
            cls._column_group_map = dict(filter(lambda kv: isinstance(kv[1], ColumnGroup), cls.__dict__.items()))
        return cls._column_group_map
