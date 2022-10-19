import copy
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO, BytesIO
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd

from pandandic import BaseFrame
from pandandic import Column


class FooBarFrame(BaseFrame):
    foo = Column(type=str)
    bar = Column(type=int)


@dataclass
class SupportedFileType:
    read_method: str
    save_method: str
    filename: str


class TestBaseFrame(TestCase):
    supported_file_types = {
        "csv": SupportedFileType(read_method="read_csv", save_method="to_csv", filename="tmp.csv"),
        "excel": SupportedFileType(read_method="read_excel", save_method="to_excel", filename="tmp.xlsx"),
        "parquet": SupportedFileType(read_method="read_parquet", save_method="to_parquet", filename="tmp.parquet")
    }

    def setUp(self) -> None:
        self.buffer = BytesIO()

    def tearDown(self) -> None:
        for supported_filetype in self.supported_file_types.values():
            if os.path.exists(supported_filetype.filename):
                os.remove(supported_filetype.filename)

    def test_should_get_typed_columns(self):
        # arrange
        sut = FooBarFrame
        expected = {
            "foo": Column(type=str),
            "bar": Column(type=int)
        }
        # act
        actual = sut.get_typed_columns()
        # assert
        self.assertEqual(expected, actual)

    def test_should_read(self):
        # arrange
        expected = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                expected.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # act
                actual = FooBarFrame().__getattribute__(file_type.read_method)(file_type.filename)
                # assert
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_omit_extra_cols(self):
        # arrange
        data = pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, None], ["two", 2, None]])
        expected = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        sut = FooBarFrame()
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                data.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # act
                actual = sut.__getattribute__(file_type.read_method)(file_type.filename)
                # assert
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_allow_extra_cols(self):
        # arrange
        expected = pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, np.nan], ["two", 2, np.nan]])
        sut = FooBarFrame().with_allowed_extra_columns(True)
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                expected.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # act
                actual = sut.__getattribute__(file_type.read_method)(file_type.filename)
                # assert
                columns = list(set(expected.columns.tolist()).union(actual.columns.tolist()))
                pd.testing.assert_frame_equal(expected[columns], actual[columns])

    def test_should_ensure_types(self):
        # arrange
        expected = pd.DataFrame(columns=["strs", "ints", "floats", "datetimes"],
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

        sut = SIFDFrame().with_validation()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                expected.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # act
                actual = sut.__getattribute__(file_type.read_method)(file_type.filename)
                # assert
                pd.testing.assert_frame_equal(expected, actual, check_dtype=True, check_column_type=True)

    def test_should_raise_on_invalid_dtype(self):
        # arrange
        data = pd.DataFrame(columns=["foo", "bar"],
                            data=[["one", "two"], ])
        sut = FooBarFrame().with_validation()
        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                data.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # assert
                with self.assertRaises(ValueError):
                    # act
                    sut.__getattribute__(file_type.read_method)(file_type.filename)

    def test_should_allow_any_dtype(self):
        # arrange
        class AnyFrame(BaseFrame):
            foo = Column(type=Any)
            bar = Column(type=object)

        data = pd.DataFrame(columns=["foo", "bar"],
                            data=[["one", 2.2], ], dtype=object)
        expected = data.copy()
        expected.loc[:, "bar"] = data.loc[:, "bar"].astype(float)
        sut = AnyFrame().with_validation()

        for name, file_type in self.supported_file_types.items():
            with self.subTest(name):
                data.__getattribute__(file_type.save_method)(file_type.filename, index=False)
                # act
                actual = sut.__getattribute__(file_type.read_method)(file_type.filename)
                # assert
                pd.testing.assert_series_equal(data.dtypes, actual.dtypes)
                actual.loc[:, "bar"] = actual.loc[:, "bar"].astype(float)
                pd.testing.assert_frame_equal(expected, actual)

    def test_should_peek_cols(self):
        # arrange
        expected_columns = ["foo", "bar", "baz", "foobar"]
        data = pd.DataFrame(columns=expected_columns,
                            data=[["one", 1, np.nan, np.nan], ["two", 2, np.nan, np.nan]])
        expected_data = data[["foo", "bar"]]
        sut = FooBarFrame().with_allowed_extra_columns(False)

        with self.subTest("csv_in_memory"):
            # arrange
            buffer = BytesIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)
            # act
            actual_columns = BaseFrame.read_csv_columns(buffer)
            actual_data = sut.read_csv(buffer)
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("csv_on_disk"):
            # arrange
            filename = self.supported_file_types["csv"].filename
            data.to_csv(filename, index=False)
            # act
            actual_columns = BaseFrame.read_csv_columns(filename)
            actual_data = sut.read_csv(filename)
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
            actual_data = sut.read_excel(buffer, engine='openpyxl')
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("excel_on_disk"):
            # arrange
            filename = self.supported_file_types["excel"].filename
            data.to_excel(filename, index=False)
            # act
            actual_columns = BaseFrame.read_excel_columns(filename)
            actual_data = sut.read_excel(filename)
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
            actual_data = sut.read_parquet(buffer)
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)

        with self.subTest("parquet_on_disk"):
            # arrange
            filename = self.supported_file_types["parquet"].filename
            data.to_parquet(filename, index=False)
            # act
            actual_columns = BaseFrame.read_parquet_columns(filename)
            actual_data = sut.read_parquet(filename)
            # assert
            self.assertListEqual(expected_columns, actual_columns)
            pd.testing.assert_frame_equal(expected_data, actual_data)
