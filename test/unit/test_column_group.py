from unittest import TestCase

from pandandic import ColumnGroup


class TestColumnGroup(TestCase):

    def test_should_implicitly_set_name_when_class_attribute(self):
        # act
        class Foo:
            bar = ColumnGroup(type=str, members=[])
        # assert
        self.assertEqual("bar", Foo.bar.name)
