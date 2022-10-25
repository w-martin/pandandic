from typing import Any
from unittest import TestCase

from pandandic import Column


class TestColumn(TestCase):

    def test_should_implicitly_set_name_when_class_attribute(self):
        # act
        class Foo:
            bar = Column(type=str)
        # assert
        self.assertEqual("bar", Foo.bar.name)

    def test_should_not_modify_type_class_attribute(self):
        # act
        foo = Column()
        foo.type = float
        bar = Column()
        # assert
        self.assertEqual(bar.type, Any)

    def test_should_not_modify_alias_class_attribute(self):
        # act
        foo = Column()
        foo.alias = "baz"
        bar = Column()
        # assert
        self.assertEqual("baz", foo.alias)
        self.assertIsNone(bar.alias)
