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

        self.assertEqual(mat[1, 1], 1)
        self.assertEqual(mat[2, 2], 2)
        self.assertEqual(mat[3, 3], 3)
        self.assertEqual(mat[4, 4], 4)

    def test_zero_indexing(self):
        values = [1, 2, 3, 4]
        row    = [1, 2, 3, 4]
        col    = [1, 2, 3, 4]
        mat = SparseMatrix(values, row, col)

        self.assertEqual(mat[0, 0], 0)
        self.assertEqual(mat[1, 2], 0)
        self.assertEqual(mat[3, 4], 0)
        self.assertEqual(mat[4, 3], 0)

    def test_out_of_range_indexing(self):
        values = [1, 2, 3, 4]
        row    = [1, 2, 3, 4]
        col    = [1, 2, 3, 4]
        mat = SparseMatrix(values, row, col)

        self.assertRaises(IndexError, mat.__getitem__, (0, -1))
        self.assertRaises(IndexError, mat.__getitem__, (-1, 0))
        self.assertRaises(IndexError, mat.__getitem__, (0, 5))
        self.assertRaises(IndexError, mat.__getitem__, (5, 0))

    def test_shape_parameter(self):
        values = [1, 2]
        row    = [0, 1]
        col    = [0, 1]
        a = SparseMatrix(values, row, col)
        b = SparseMatrix(values, row, col, shape=(3, 4))
        c = SparseMatrix([1, 1], [12, 23], [7, 9])

        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(b.shape, (3, 4))
        self.assertRaises(IndexError, a.__getitem__, (2, 3))
        self.assertEqual(b[2, 3], 0)
        self.assertEqual(c.shape, (24, 10))

    def test_equals(self):
        a = SparseMatrix([1, 1, 1],
                         [0, 1, 2],
                         [0, 1, 2])
        
        b = SparseMatrix([1, 1, 1], 
                         [2, 1, 0], 
                         [2, 1, 0])
        
        c = SparseMatrix([1, 1, 1], 
                         [2, 0, 1], 
                         [2, 0, 1])
        
        d = SparseMatrix([1, 1, 1, 0], 
                         [0, 1, 2, 1], 
                         [0, 1, 2, 2])
        
        e = SparseMatrix([1, 1, 1],
                         [0, 1, 2],
                         [0, 1, 2], shape=(4, 4))

        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(b, c)
        self.assertEqual(a, d)
        self.assertNotEqual(a, e)
        
        

    def test_add(self):
        a = SparseMatrix([1, 1, 1], [0, 1, 2], [0, 1, 2], shape=(3, 3))
        b = SparseMatrix([1, 1, 1], [1, 1, 1], [0, 1, 2], shape=(3, 3))
        c = SparseMatrix([1, 1, 2, 1, 1], 
                         [0, 1, 1, 1, 2],
                         [0, 0, 1, 2, 2])
        
        d = SparseMatrix([1, -1, -1, 1], 
                         [0, 1, 1, 2],
                         [0, 0, 2, 2], shape=(3,3))
        
        #e = SparseMatrix([], 
        #                 [], 
        #                 [], shape=(3, 3))
        
        self.assertEqual(c, a+b)
        print(a)
        print(b)
        print(a-b)
        print(d)
        self.assertEqual(d, a-b)
        
        
        
        e = SparseMatrix([], [], [], shape=(10, 10))
        self.assertRaises(ArithmeticError, a.__add__, e)
        
    def test_radd(self):
        a = SparseMatrix([1, 1, 1], [0, 1, 2], [0, 1, 2], shape=(3, 3))
        b = SparseMatrix([1, 1, 1], [1, 1, 1], [0, 1, 2], shape=(3, 3))
        c = SparseMatrix([1, 1, 2, 1, 1], 
                         [0, 1, 1, 1, 2],
                         [0, 0, 1, 2, 2])

        self.assertEqual(a, 0+a)
        
        d = sum([a, b])
        
        self.assertEqual(c, d)
        

    def test_transposed(self):
        a = SparseMatrix([1, 1, 1], [0, 1, 2], [0, 1, 2])
        self.assertEqual(a, a.T)
        
        b = SparseMatrix([1, 1, 1], [0, 0, 0], [0, 1, 2])
        c = SparseMatrix([1, 1, 1], [0, 1, 2], [0, 0, 0])
        self.assertEqual(b, c.T)
    
    def test_setitem(self):
        a = SparseMatrix([1, 1, 1], [0, 1, 2], [0, 1, 2])
        self.assertEqual(a[0,0], 1)
        a[0,0] = 5
        self.assertEqual(a[0,0], 5)
        
        self.assertEqual(a[0,2], 0)
        a[0,2] = 5
        self.assertEqual(a[0,2], 5)
        
        self.assertRaises(IndexError, a.__setitem__, (0, 3), 5)
        
    def test_matmul(self):
        # test shape
        a = SparseMatrix([], [], [], shape=(3, 5))
        b = SparseMatrix([], [], [], shape=(5, 7))
        self.assertEqual((3, 7), (a @ b).shape)
        
        
        #example from Wikipedia
        a = SparseMatrix([3, 2, 1, 1, 2],
                         [0, 0, 0, 1, 1],
                         [0, 1, 2, 0, 2])
        b = SparseMatrix([1, 2, 1, 4],
                         [0, 0, 1, 2],
                         [0, 1, 1, 0])
        c = SparseMatrix([7, 8, 9, 2], 
                         [0, 0, 1, 1], 
                         [0, 1, 0, 1])
        
        self.assertEqual(c, a @ b)
        
    def test_mul(self):
        a = SparseMatrix([1, 2, 3], [0, 1, 0], [1, 2, 0])
        b = 2
        c = SparseMatrix([2, 4, 6], [0, 1, 0], [1, 2, 0])
        
        self.assertEqual(a*b, c)
        print(b*a)
        self.assertEqual(b*a, c)

if __name__ == '__main__':
    unittest.main()