import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Any, Callable
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame

from pandandic import BaseFrame, ColumnSet, DefinedLater, ColumnGroup
from pandandic import Column
from pandandic.column_alias_not_yet_defined_exception import ColumnAliasNotYetDefinedException
from pandandic.column_group_exception import ColumnGroupException
from pandandic.column_set_members_not_yet_defined_exception import ColumnSetMembersNotYetDefinedException


class FooBarFrame(BaseFrame):
    foo = Column(type=str)
    bar = Column(type=int)


@dataclass
class SupportedFileType:
    read_method: Callable
    save_method: Callable
    filename: Callable


def to_avro(df, filename, *args, **kwargs):
    if "index" in kwargs:
        del kwargs["index"]
    from pandavro import to_avro
    avro_type_dict = {
        str: 'string',
        object: 'string',
        type(None): 'string',
        int: 'int',
        float: 'double',
        pd._libs.tslibs.timestamps.Timestamp: {'type': 'long', 'logicalType': 'timestamp-micros'}
    }
    schema = {'type': 'record', 'name': 'Root',
              'fields': [{'name': k, 'type': ['null', avro_type_dict[type(v)]]} for k, v in
                         df.iloc[0].to_dict().items()]}
    for column, dtype in df.dtypes.items():
        if str(dtype) == "datetime64[ns]":
            df = df.copy()
            df.loc[:, column] = df.loc[:, column].dt.tz_localize("utc")
    to_avro(filename, df, *args, schema=schema, **kwargs)


class DfStore:
    def __init__(self):
        self._df_in_flight = None

    def get_df_in_flight(self):
        return self._df_in_flight

    def set_df_in_flight(self, df):
        self._df_in_flight = df


store = DfStore()


def to_df(df, *args, **kwargs):
    store.set_df_in_flight(df)


class TestBaseFrame(TestCase):
    supported_file_types = {
        "csv": SupportedFileType(read_method=BaseFrame.read_csv, save_method=DataFrame.to_csv,
                                 filename=lambda: "tmp.csv"),
        "excel": SupportedFileType(read_method=BaseFrame.read_excel, save_method=DataFrame.to_excel,
                                   filename=lambda: "tmp.xlsx"),
        "parquet": SupportedFileType(read_method=BaseFrame.read_parquet, save_method=DataFrame.to_parquet,
                                     filename=lambda: "tmp.parquet"),
        "avro": SupportedFileType(read_method=BaseFrame.read_avro, save_method=to_avro,
                                  filename=lambda: "tmp.avro"),
        "df": SupportedFileType(read_method=BaseFrame.from_df, save_method=to_df,
                                filename=store.get_df_in_flight)
    }

    def setUp(self) -> None:
        self.buffer = BytesIO()

    def tearDown(self) -> None:
        for supported_filetype in self.supported_file_types.values():
            if isinstance(supported_filetype.filename(), str) and os.path.exists(supported_filetype.filename()):
                os.remove(supported_filetype.filename())

    def test_should_get_typed_columns(self):
        # arrange
        sut = FooBarFrame
        expected = {
            "foo": Column(type=str),
            "bar": Column(type=int)
        }
        # act
        actual = sut._get_column_map()
        # assert
        self.assertEqual(expected, actual)

    def test_should_read(self):
        # arrange
        expected = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        sut = FooBarFrame()
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                actual = file_type.read_method(sut, file_type.filename()).to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_omit_extra_cols(self):
        # arrange
        data = pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, None], ["two", 2, None]])
        expected = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        sut = FooBarFrame()
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # act
                actual = file_type.read_method(sut, file_type.filename()).to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_allow_extra_cols(self):
        # arrange
        expected = pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, np.nan], ["two", 2, np.nan]])
        sut = FooBarFrame().with_extra_columns_allowed(True)
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                actual = file_type.read_method(sut, file_type.filename()).to_df()
                # assert
                columns = list(set(expected.columns.tolist()).union(actual.columns.tolist()))
                pd.testing.assert_frame_equal(expected[columns], actual[columns])

    def test_should_ensure_types(self):
        # arrange
        data = pd.DataFrame(columns=["strs", "ints", "floats", "datetimes"],
                            data=[
                                ["one", 1, 1., datetime(1970, 1, 1)],
                                ["two", 2, 2., datetime(1970, 2, 2)],
                                ["three", 3, 3., datetime(1970, 3, 3)],
                            ])

        class SIFDFrame(BaseFrame):
            strs = Column(type=str)
            ints = Column(type=int)
            floats = Column(type=float)
            datetimes = Column(type=datetime)

        sut = SIFDFrame().with_enforced_types()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                expected = data
                # act
                actual = file_type.read_method(sut, file_type.filename()).to_df()
                # assert
                if str(actual["datetimes"].dtype) == "datetime64[ns, UTC]":
                    expected = data.copy()
                    expected.loc[:, "datetimes"] = expected.loc[:, "datetimes"].dt.tz_localize("utc")
                pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_column_type=True)

    def test_should_raise_on_invalid_dtype(self):
        # arrange
        data = pd.DataFrame(columns=["foo", "bar"],
                            data=[["one", "two"], ])
        sut = FooBarFrame().with_enforced_types()
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # assert
                with self.assertRaises(ValueError):
                    # act
                    file_type.read_method(sut, file_type.filename())

    def test_should_allow_any_dtype(self):
        # arrange
        class AnyFrame(BaseFrame):
            foo = Column(type=Any)
            bar = Column(type=object)

        data = pd.DataFrame(columns=["foo", "bar"],
                            data=[["one", 2.2], ], dtype=object)
        expected = data.copy()
        expected.loc[:, "bar"] = data.loc[:, "bar"].astype(float)
        sut = AnyFrame().with_enforced_types()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # act
                actual = file_type.read_method(sut, file_type.filename()).to_df()
                # assert
                pd.testing.assert_series_equal(data.dtypes, actual.dtypes)
                actual.loc[:, "bar"] = actual.loc[:, "bar"].astype(float)
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_fetch_exact_column_groups(self):
        # arrange
        class GroupFrame(BaseFrame):
            foo = ColumnSet(type=float, members=["foo-0", "foo-1"])
            bar = ColumnSet(type=str, members=["bar", "bartoo"])

        expected = pd.DataFrame(columns=["foo-0", "foo-1", "bar", "bartoo"],
                                data=[[2.2, 2.1, "a", "b"], [3.2, 3.1, "c", "d"]])
        sut = GroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)
                pd.testing.assert_frame_equal(expected[["foo-0", "foo-1"]], result.foo)
                pd.testing.assert_frame_equal(expected[GroupFrame.bar.members], result.bar)

    def test_should_fetch_regex_column_groups(self):
        # arrange
        class GroupFrame(BaseFrame):
            foo = ColumnSet(type=float, members=["foo-\d+"], regex=True)
            bar = ColumnSet(type=str, members=["bar*"], regex=True)

        expected = pd.DataFrame(columns=["foo-0", "foo-1", "bar", "bartoo"],
                                data=[[2.2, 2.1, "a", "b"], [3.2, 3.1, "c", "d"]])
        sut = GroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)
                pd.testing.assert_frame_equal(expected[["foo-0", "foo-1"]], result.foo)
                pd.testing.assert_frame_equal(expected[["bar", "bartoo"]], result.bar)

    def test_should_raise_on_conflicting_column_groups(self):
        # arrange
        class GroupFrame(BaseFrame):
            foo = ColumnSet(type=float, members=["foo-\d+"], regex=True)
            bar = ColumnSet(type=str, members=["bar", "bartoo", "foo-0"], regex=False)

        expected = pd.DataFrame(columns=["foo-0", "foo-1", "bar", "bartoo"],
                                data=[[2.2, 2.1, "a", "b"], [3.2, 3.1, "c", "d"]])
        sut = GroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # assert
                with self.assertRaises(ColumnGroupException):
                    # act
                    file_type.read_method(sut, file_type.filename())

    def test_should_allow_conflicting_column_groups_if_greedy(self):
        # arrange
        class GroupFrame(BaseFrame):
            foo = ColumnSet(type=float, members=["foo-\d+", "bar"], regex=True)
            bar = ColumnSet(type=str, members=["bar", "bartoo"], regex=False)

        expected = pd.DataFrame(columns=["foo-0", "foo-1", "bar", "bartoo"],
                                data=[[2.2, 2.1, "a", "b"], [3.2, 3.1, "c", "d"]])
        sut = GroupFrame().with_greedy_column_sets()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)
                pd.testing.assert_frame_equal(expected[["foo-0", "foo-1"]], result.foo)
                pd.testing.assert_frame_equal(expected[["bar", "bartoo"]], result.bar)

    def test_should_fetch_columns_and_column_groups(self):
        # arrange
        class GroupFrame(BaseFrame):
            foo = Column(type=float)
            bar = ColumnSet(type=str, members=["bar.*"], regex=True)
            baz = ColumnSet(type=int, members=["baz-0", "baz-1"], regex=False)

        expected = pd.DataFrame(columns=["foo", "bar", "bartoo", "baz-0", "baz-1"],
                                data=[[2.2, "a", "b", 21, 22], [3.2, "c", "d", 31, 32]])
        sut = GroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(expected, actual)
                pd.testing.assert_series_equal(expected["foo"], result.foo)
                pd.testing.assert_frame_equal(expected[["bar", "bartoo"]], result.bar)
                pd.testing.assert_frame_equal(expected[GroupFrame.baz.members], result.baz)

    def test_should_peek_cols(self):
        # arrange
        expected_columns = ["foo", "bar", "baz", "foobar"]
        data = pd.DataFrame(columns=expected_columns,
                            data=[["one", 1, np.nan, np.nan], ["two", 2, np.nan, np.nan]])
        expected_data = data[["foo", "bar"]]
        sut = FooBarFrame().with_extra_columns_allowed(False)

        with self.subTest("csv_in_memory"):
            # arrange
            buffer = BytesIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)
            # act
            actual_columns = BaseFrame.read_csv_columns(buffer)
            actual_data = sut.read_csv(buffer).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("csv_on_disk"):
            # arrange
            filename = self.supported_file_types["csv"].filename()
            data.to_csv(filename, index=False)
            # act
            actual_columns = BaseFrame.read_csv_columns(filename)
            actual_data = sut.read_csv(filename).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("excel_in_memory"):
            # arrange
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False)
            buffer.seek(0)
            # act
            actual_columns = BaseFrame.read_excel_columns(buffer, engine='openpyxl')
            actual_data = sut.read_excel(buffer, engine='openpyxl').to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("excel_on_disk"):
            # arrange
            filename = self.supported_file_types["excel"].filename()
            data.to_excel(filename, index=False)
            # act
            actual_columns = BaseFrame.read_excel_columns(filename)
            actual_data = sut.read_excel(filename).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("parquet_in_memory"):
            # arrange
            buffer = BytesIO()
            data.to_parquet(buffer)
            buffer.seek(0)
            # act
            actual_columns = BaseFrame.read_parquet_columns(buffer)
            actual_data = sut.read_parquet(buffer).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("parquet_on_disk"):
            # arrange
            filename = self.supported_file_types["parquet"].filename()
            data.to_parquet(filename, index=False)
            # act
            actual_columns = BaseFrame.read_parquet_columns(filename)
            actual_data = sut.read_parquet(filename).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("avro_in_memory"):
            # arrange
            buffer = BytesIO()
            to_avro(data, buffer)
            buffer.seek(0)
            # act
            actual_columns = BaseFrame.read_avro_columns(buffer)
            actual_data = sut.read_avro(buffer).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("avro_on_disk"):
            # arrange
            filename = self.supported_file_types["avro"].filename()
            to_avro(data, filename)
            # act
            actual_columns = BaseFrame.read_avro_columns(filename)
            actual_data = sut.read_avro(filename).to_df()
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

    def test_should_read_dynamic_column(self):
        # arrange
        class DynamicColumnFrame(BaseFrame):
            foo = Column(type=int)
            bar = Column()

        foo_column_name = "foob"
        bar_column_name = "barb"

        data = pd.DataFrame(
            columns=[foo_column_name, bar_column_name],
            data=[
                [20, "baz"],
                [21, "17.5"],
                [21, "36"],
            ]
        )
        sut = DynamicColumnFrame()
        DynamicColumnFrame.foo.alias = foo_column_name
        DynamicColumnFrame.bar.alias = bar_column_name

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(data, actual)
                pd.testing.assert_series_equal(data[foo_column_name], result.foo)
                pd.testing.assert_series_equal(data[bar_column_name], result.bar)

    def test_should_raise_if_column_not_defined_yet(self):
        # arrange
        class DynamicColumnFrame(BaseFrame):
            foo = Column(alias=DefinedLater)

        foo_column_name = "foob"
        bar_column_name = "barb"

        data = pd.DataFrame(
            columns=[foo_column_name, bar_column_name],
            data=[
                [20, "baz"],
                [21, "17.5"],
                [21, "36"],
            ]
        )
        sut = DynamicColumnFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # assert
                with self.assertRaises(ColumnAliasNotYetDefinedException):
                    # act
                    file_type.read_method(sut, file_type.filename())
                # assert
                with self.assertRaises(ColumnAliasNotYetDefinedException):
                    # act
                    _ = sut.foo

    def test_should_read_dynamic_column_set(self):
        # arrange
        class DynamicColumnFrame(BaseFrame):
            foo = ColumnSet(members=DefinedLater)
            bar = ColumnSet(type=int, members=DefinedLater)

        foo_column_names = ["foob", "fooc", "food"]
        bar_column_names = ["barb", "barc"]

        data = pd.DataFrame(
            columns=foo_column_names + bar_column_names,
            data=[
                [20, "baz", 1, 2, 3],
                [21, "17.5", 1, 2, 3],
                [21, "36", 1, 2, 3],
            ]
        )
        sut = DynamicColumnFrame()
        DynamicColumnFrame.foo.members = foo_column_names
        DynamicColumnFrame.bar.members = bar_column_names

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                actual = result.to_df()
                # assert
                pd.testing.assert_frame_equal(data, actual)
                pd.testing.assert_frame_equal(data[foo_column_names], result.foo)
                pd.testing.assert_frame_equal(data[bar_column_names], result.bar)

    def test_should_raise_if_column_set_not_defined_yet(self):
        # arrange
        class DynamicColumnFrame(BaseFrame):
            foo = ColumnSet(members=DefinedLater)
            bar = ColumnSet(type=int, members=DefinedLater)

        foo_column_names = ["foob", "fooc", "food"]
        bar_column_names = ["barb", "barc"]

        data = pd.DataFrame(
            columns=foo_column_names + bar_column_names,
            data=[
                [20, "baz", 1, 2, 3],
                [21, "17.5", 1, 2, 3],
                [21, "36", 1, 2, 3],
            ]
        )
        sut = DynamicColumnFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(data, file_type.filename(), index=False)
                # assert
                with self.assertRaises(ColumnSetMembersNotYetDefinedException):
                    # act
                    file_type.read_method(sut, file_type.filename())
                # assert
                with self.assertRaises(ColumnSetMembersNotYetDefinedException):
                    # act
                    _ = sut.foo

    def test_should_enforce_column_set_column_order_when_accessed_by_attribute(self):
        # arrange

        class OrderedGroupFrame(BaseFrame):
            foo = Column(type=float)
            bar = ColumnSet(type=str, members=["bar.*"], regex=True)
            baz = ColumnSet(type=int, members=["baz-0", "baz-1"], regex=False)

        expected = pd.DataFrame(columns=["foo", "bar", "bartoo", "baz-1", "baz-0"],
                                data=[[2.2, "a", "b", 21, 22],
                                      [3.2, "c", "d", 31, 32]])

        sut = OrderedGroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                # assert
                pd.testing.assert_frame_equal(expected[OrderedGroupFrame.baz.members], result.baz)

    def test_should_use_column_alias_within_column_group(self):
        # arrange

        class AliasColumInGroupFrame(BaseFrame):
            foo = Column(type=float)
            bar = Column(alias='rab')
            baz = Column(type=str)
            spam = ColumnGroup(members=[bar, baz])

        expected = pd.DataFrame(columns=["foo", "rab", "baz"],
                                data=[[2.2, "a", "b"],
                                      [3.2, "c", "d"]])

        sut = AliasColumInGroupFrame()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                file_type.save_method(expected, file_type.filename(), index=False)
                # act
                result = file_type.read_method(sut, file_type.filename())
                # assert
                pd.testing.assert_frame_equal(expected[[AliasColumInGroupFrame.bar.alias,
                                                       AliasColumInGroupFrame.baz.name]], result.spam)
