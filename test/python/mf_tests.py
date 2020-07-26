import unittest
import mf


class MyTestCase(unittest.TestCase):
    def test_say_hello(self):
        name = 'Joe'
        self.assertEqual(mf.say_hello(name), 'Hello, Joe!')


if __name__ == '__main__':
    unittest.main()
