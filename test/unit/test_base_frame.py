import os
from unittest import TestCase

import pandas as pd

from src.base_frame import BaseFrame
from src.column import Column


class FooBarFrame(BaseFrame):
    foo = Column(type=str)
    bar = Column(type=int)


class TestBaseFrame(TestCase):
    filename: str

    fixture: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        cls.filename = "TestBaseFrame.csv"
        cls.fixture = pd.DataFrame(columns=["foo", "bar"], data=[["one", 1], ["two", 2]])
        cls.fixture.to_csv(cls.filename, index=False)

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.filename)

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
        # arrange

        # act
        sut = FooBarFrame.from_csv(self.filename)
        # assert
        pd.testing.assert_frame_equal(self.fixture, sut)
