import numpy as np


class SparseMatrix:
    """
    Data structure for sparse matrices.
    
    Parameter
    ---------
    data : 1D-array
        Values of elements.
    row : 1D-array of integers.
        Row indices of elements.
    col : 1D-array of integers.
        Col indices of elements.
        
    Attributes
    ----------
    data : 1D-array
        Values of elements.
    row : 1D-array of integers.
        Row indices of elements.
    col : 1D-array of integers.
        Col indices of elements.
    """
    
    def __init__(self, data, row, col, size=None):
        
        #### validate the input ####

        # input lists same length?
        if not (len(data) == len(row) == len(col)):
            raise ValueError('data, row and col must have the same length!')

        # no negative indizes?
        if any([ x<0 or y<0 for x, y in zip(row, col)]):
            raise ValueError('indices must be positive!')
        # correct size?
        if size == None:
            self.size = (max(row)+1, max(col)+1)
        else:
            assert len(size) == 2
            assert type(size[0]) == type(size[1]) == int
            assert size[0] > 0
            assert size[1] > 0

            self.size = size

        self.data = data
        self.row = row
        self.col = col

        self.content = dict()
        for d, r, c in zip(data, row, col):
            self.content[(r, c)] = d
    
    def get(self, r, c):
        if not ((0 <= r <= self.size[0]-1) and (0 <= c <= self.size[1]-1)):
            raise IndexError('Index out of matrix size')
        
        return self.content.get((r, c), 0)
    

    def __repr__(self) -> str:
        return str(self.content)

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass


if __name__ == '__main__':
    mat = SparseMatrix()
    print(type(mat))