from typing import Collection
import unittest
from unittest import main
from sparse import SparseMatrix


class TestSparse(unittest.TestCase):

    def test_input_same_length(self):
        self.assertRaises(ValueError, SparseMatrix, [1, 2], [1], [1])
        self.assertRaises(ValueError, SparseMatrix, [1], [1, 2], [1])
        self.assertRaises(ValueError, SparseMatrix, [1], [1], [1, 2])

    def test_no_negative_indices(self):
        self.assertRaises(ValueError, SparseMatrix, [1, 2], [1, -2], [1, 2])
        self.assertRaises(ValueError, SparseMatrix, [1, 2], [1, 2], [1, -2])

    def test_non_zero_indexing(self):
        values = [1, 2, 3, 4]
        row    = [1, 2, 3, 4]
        col    = [1, 2, 3, 4]
        mat = SparseMatrix(values, row, col)

        self.assertEqual(mat.get(1, 1), 1)
        self.assertEqual(mat.get(2, 2), 2)
        self.assertEqual(mat.get(3, 3), 3)
        self.assertEqual(mat.get(4, 4), 4)

    def test_zero_indexing(self):
        values = [1, 2, 3, 4]
        row    = [1, 2, 3, 4]
        col    = [1, 2, 3, 4]
        mat = SparseMatrix(values, row, col)

        self.assertEqual(mat.get(0, 0), 0)
        self.assertEqual(mat.get(1, 2), 0)
        self.assertEqual(mat.get(3, 4), 0)
        self.assertEqual(mat.get(4, 3), 0)

    def test_out_of_range_indexing(self):
        values = [1, 2, 3, 4]
        row    = [1, 2, 3, 4]
        col    = [1, 2, 3, 4]
        mat = SparseMatrix(values, row, col)

        self.assertRaises(IndexError, mat.get, 0, -1)
        self.assertRaises(IndexError, mat.get, -1, 0)
        self.assertRaises(IndexError, mat.get, 0, 5)
        self.assertRaises(IndexError, mat.get, 5, 0)

    def test_size_parameter(self):
        values = [1, 2]
        row    = [0, 1]
        col    = [0, 1]
        a = SparseMatrix(values, row, col)
        b = SparseMatrix(values, row, col, size=(3, 4))
        c = SparseMatrix([1, 1], [12, 23], [7, 9])

        self.assertEqual(a.size, (2, 2))
        self.assertEqual(b.size, (3, 4))
        self.assertRaises(IndexError, a.get, 2, 3)
        self.assertEqual(b.get(2, 3), 0)
        self.assertEqual(c.size, (24, 10))


if __name__ == '__main__':
    unittest.main()