import itertools
import re
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, date
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from pandas import DataFrame

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


class BaseFrame(DataFrame):
    """
    Enabled schema-in-code through subclassing of DataFrame.
    Wraps pandas read functions in order to apply the defined schema.
    See https://github.com/w-martin/pandandic for usage documentation or to submit bug reports.
    """
    _column_map: Dict[str, Column] = None
    _column_set_map: Dict[str, ColumnSet] = None
    _column_group_map: Dict[str, ColumnGroup] = None
    _column_consumed_map: Dict[str, List[str]] = defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enforce_types = True
        self.enforce_columns = True
        self.allow_extra_columns = False
        self.greedy_column_sets = False

        self._get_column_map()
        self._get_column_set_map()
        self._get_column_group_map()

        if len(args) > 0 and isinstance(args[0], DataFrame):
            self.from_df(args[0])

    def __getattribute__(self, item):
        if not item.startswith("_"):
            if self._column_map is not None and item in self._column_map:
                column = self._column_map[item]
                if column.alias == DefinedLater or isinstance(column.alias, DefinedLater):
                    raise ColumnAliasNotYetDefinedException(column)
                return self[column.alias or column.name]

            if self._column_set_map is not None and item in self._column_set_map:
                column_set = self._column_set_map[item]
                if column_set.members == DefinedLater or isinstance(column_set.members, DefinedLater):
                    raise ColumnSetMembersNotYetDefinedException(column_set)
                consumed_cols = self._column_consumed_map[column_set.name]
                if not column_set.regex:
                    consumed_cols = [c for c in column_set.members if c in consumed_cols]
                return self[consumed_cols]

            if self._column_group_map is not None and item in self._column_group_map:
                return self[list(itertools.chain.from_iterable(
                    [[column_or_column_group.name]
                     if isinstance(column_or_column_group, Column)
                     else self._column_consumed_map[column_or_column_group.name]
                     for column_or_column_group in self._column_group_map[item].members]))]

        return super().__getattribute__(item)

    def to_df(self) -> DataFrame:
        return DataFrame(self)

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

    def read_csv(self, *args, **kwargs) -> Self:
        if self.enforce_columns or self.enforce_types:
            columns = self.read_csv_columns(*args, **kwargs)
            column_map = self._compute_column_map(columns)

            if self.enforce_columns:
                allowed_columns = list(column_map.keys())
                if self.allow_extra_columns:
                    allowed_columns = list(set(allowed_columns).union(columns))

                kwargs["usecols"] = allowed_columns

            if self.enforce_types:
                self._apply_validation(column_map, kwargs)

        df = pd.read_csv(*args, **kwargs)
        super().__init__(df)
        return self

    def read_excel(self, *args, **kwargs) -> Self:
        if self.enforce_columns or self.enforce_types:
            columns = self.read_excel_columns(*args, **kwargs)
            column_map = self._compute_column_map(columns)

            if self.enforce_columns:
                allowed_columns = list(column_map.keys())
                if self.allow_extra_columns:
                    allowed_columns = list(set(allowed_columns).union(columns))

                kwargs["usecols"] = allowed_columns

            if self.enforce_types:
                self._apply_validation(column_map, kwargs)

        df = pd.read_excel(*args, **kwargs)
        super().__init__(df)
        return self

    def read_parquet(self, *args, **kwargs) -> Self:
        columns = self.read_parquet_columns(*args, **kwargs)
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(column_map.keys())
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            kwargs["columns"] = allowed_columns

        df = pd.read_parquet(*args, **kwargs)
        for column, t in filter(self._type_is_castable, column_map.items()):
            df.loc[:, column] = df.loc[:, column].astype(t)
        super().__init__(df)
        return self

    def read_avro(self, *args, **kwargs) -> Self:
        from pandavro import read_avro

        columns = self.read_avro_columns(*args, **kwargs)
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(column_map.keys())
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            kwargs["columns"] = allowed_columns

        df = read_avro(*args, **kwargs)
        for column, t in filter(self._type_is_castable, column_map.items()):
            df.loc[:, column] = df.loc[:, column].astype(t)
        super().__init__(df)
        return self

    def from_df(self, df) -> Self:
        df = df.copy()

        columns = df.columns.tolist()
        column_map = self._compute_column_map(columns)

        if self.enforce_columns:
            allowed_columns = list(column_map.keys())
            if self.allow_extra_columns:
                allowed_columns = list(set(allowed_columns).union(columns))

            df = df[allowed_columns]

        for column, t in filter(self._type_is_castable, column_map.items()):
            df.loc[:, column] = df.loc[:, column].astype(t)
        super().__init__(df)
        return self

    def _compute_column_map(self, columns: List[str]) -> Dict[str, type]:
        self._column_consumed_map.clear()

        key_column_map = {(column.alias or column.name): column for column in self._get_column_map().values()}
        for alias, column in key_column_map.items():
            if alias == DefinedLater or isinstance(alias, DefinedLater):
                raise ColumnAliasNotYetDefinedException(column)

        if len(self._get_column_set_map()) == 0:
            return {k: v.type for k, v in key_column_map.items()}

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
                        self._column_consumed_map[column_set.name].append(column)

            for column_set in regex_column_sets:
                if any((re.match(member, column) for member in column_set.members)):
                    if consumed_columns[i]:
                        if not self.greedy_column_sets:
                            raise ColumnGroupException(column, column_bag[i], column_set)
                    else:
                        consumed_columns[i] = True
                        column_bag[i] = column_set
                        self._column_consumed_map[column_set.name].append(column)

        result = {columns[i]: column_or_group.type
                  for i, column_or_group in
                  filter(lambda ix: ix[1] is not None, enumerate(column_bag))}
        if self.enforce_columns:
            for column_set in exact_column_sets:
                for column in column_set.members:
                    if column not in result:
                        result[column] = column_set.type
        return result

    def _apply_validation(self, column_map: Dict[str, type], kwargs: Dict[str, Any]):
        kwargs["dtype"] = dict(
            filter(self._type_is_not_date,
                   filter(self._type_is_not_any,
                          column_map.items())))
        kwargs["parse_dates"] = list(
            map(lambda t: t[0],
                filter(self._type_is_date, column_map.items())))

    @staticmethod
    def _type_is_castable(column_type_tuple: Tuple[str, type]) -> bool:
        return BaseFrame._type_is_not_any(column_type_tuple) and BaseFrame._type_is_not_date(column_type_tuple)

    @staticmethod
    def _type_is_not_any(column_type_tuple: Tuple[str, type]) -> bool:
        _, t = column_type_tuple
        return t not in (Any, None)

    @staticmethod
    def _type_is_date(column_type_tuple: Tuple[str, type]) -> bool:
        _, t = column_type_tuple
        return t in (datetime, "datetime", date, "date")

    @staticmethod
    def _type_is_not_date(column_type_tuple: Tuple[str, type]) -> bool:
        return not BaseFrame._type_is_date(column_type_tuple)

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

    @staticmethod
    def read_avro_columns(filepath_or_buffer, *args, **kwargs) -> list:
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
