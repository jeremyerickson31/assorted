import unittest
from utils import simple_functions


class TestSimpleFunction(unittest.TestCase):

    def test_add_two_integers(self):
        z = simple_functions.add_two_inputs(1, 2)
        self.assertEqual(z, 3)

    def test_add_two_strings(self):
        z = simple_functions.add_two_inputs("Hello", "John")
        self.assertEqual(z, "Hello John")


if __name__ == '__main__':
    unittest.main()
