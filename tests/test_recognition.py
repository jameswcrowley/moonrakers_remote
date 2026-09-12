# unit tests for recognition.py:
import src.recognition as recognition
import unittest

class TestRecognition(unittest.TestCase):
    def test_example(self):
        self.assertTrue(True)

    def test_recognition_instance(self):
        SIFT = recognition.SIFTMatcher()
        self.assertIsInstance(SIFT, recognition.SIFTMatcher)
        self.assertTrue(hasattr(SIFT, 'match'))
        self.assertTrue(callable(getattr(SIFT, 'match', None)))

if __name__ == '__main__':
    unittest.main()