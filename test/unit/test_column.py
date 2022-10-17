from unittest import TestCase

from pandantic import Column


class TestColumn(TestCase):

    def test_should_implicitly_set_name_when_class_attribute(self):
        # act
        class Foo:
            bar = Column(type=str)
        # assert
        self.assertEqual("bar", Foo.bar.name)
