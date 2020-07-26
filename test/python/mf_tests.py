import unittest
import mf


class MyTestCase(unittest.TestCase):
    def test_say_hello(self):
        name = 'Joe'
        self.assertEqual(mf.say_hello(name), 'Hello, Joe!')

    def test_square_matrix(self):
        n = 3
        m = mf.square_matrix(n)
        self.assertEqual(m.shape, (3,3))


if __name__ == '__main__':
    unittest.main()
