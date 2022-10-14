import os
from io import BytesIO
from unittest import TestCase

import numpy as np
import pandas as pd

from pandantic import BaseFrame
from pandantic import Column


class FooBarFrame(BaseFrame):
    foo = Column(type=str)
    bar = Column(type=int)


class TestBaseFrame(TestCase):
    buffer: BytesIO
    fixture: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        cls.buffer = BytesIO()
        cls.fixture.to_csv(cls.buffer, index=False)
        cls.temp_csv = "tmp.csv"

    def setUp(self) -> None:
        self.buffer.seek(0)

    def tearDown(self) -> None:
        if os.path.exists(self.temp_csv):
            os.remove(self.temp_csv)

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

    def test_should_read_csv(self):
        # act
        actual = FooBarFrame().from_csv(self.buffer)
        # assert
        pd.testing.assert_frame_equal(self.fixture, actual)

    def test_should_omit_extra_cols_from_csv(self):
        # arrange
        buffer = BytesIO()
        pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, None], ["two", 2, None]]).to_csv(buffer)
        buffer.seek(0)
        # act
        actual = FooBarFrame().from_csv(buffer)
        # assert
        pd.testing.assert_frame_equal(self.fixture, actual)

    def test_should_allow_extra_cols_from_csv(self):
        # arrange
        buffer = BytesIO()
        expected = pd.DataFrame(columns=["foo", "bar", "baz"], data=[["one", 1, np.nan], ["two", 2, np.nan]])
        expected.to_csv(buffer, index=False)
        buffer.seek(0)
        # act
        actual = FooBarFrame().with_allowed_extra_columns().from_csv(buffer)
        # assert
        pd.testing.assert_frame_equal(expected, actual)

    def test_should_peek_cols_from_csv_in_memory(self):
        # arrange
        buffer = BytesIO()
        expected = pd.DataFrame(columns=["foo", "bar", "baz", "foobar"],
                                data=[["one", 1, np.nan, np.nan], ["two", 2, np.nan, np.nan]])
        expected.to_csv(buffer, index=False)
        buffer.seek(0)
        # act
        columns = BaseFrame.peek_columns_from_csv(buffer, usecols=["foo", "bar"])
        data = FooBarFrame().from_csv(buffer, usecols=["foo", "bar"])
        # assert
        self.assertListEqual(["foo", "bar", "baz", "foobar"], columns.tolist())
        pd.testing.assert_frame_equal(expected[["foo", "bar"]], data)

    def test_should_peek_cols_from_csv_on_disk(self):
        # arrange
        expected = pd.DataFrame(columns=["foo", "bar", "baz", "foobar"],
                                data=[["one", 1, np.nan, np.nan], ["two", 2, np.nan, np.nan]])
        expected.to_csv(self.temp_csv, index=False)
        # act
        columns = BaseFrame.peek_columns_from_csv(self.temp_csv, usecols=["foo", "bar"])
        data = FooBarFrame().from_csv(self.temp_csv, usecols=["foo", "bar"])
        # assert
        self.assertListEqual(["foo", "bar", "baz", "foobar"], columns.tolist())
        pd.testing.assert_frame_equal(expected[["foo", "bar"]], data)
