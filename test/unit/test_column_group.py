from unittest import TestCase

from pandandic import ColumnSet


class TestColumnGroup(TestCase):

    def test_should_implicitly_set_name_when_class_attribute(self):
        # act
        class Foo:
            bar = ColumnSet(type=str, members=[])
        # assert
        self.assertEqual("bar", Foo.bar.name)
