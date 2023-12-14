import unittest
import numpy as np

from emg_classifier.emg_stream import fifo_buffer


class TestEmgStream(unittest.TestCase):

    def test_fifo_buffer(self):
        mock_buffer = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
        new_samples = np.array([[4, 5, 6], [4, 5, 6]])

        expected_outcome = np.array([[3, 4, 5, 6], [3, 4, 5, 6]])

        fifo_buffer(mock_buffer, new_samples)

        self.assertEqual(expected_outcome.tolist(), mock_buffer.tolist())


if __name__ == '__main__':
    unittest.main()
