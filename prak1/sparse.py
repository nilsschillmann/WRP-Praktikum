import numpy as np

# TODO: function documention
# TODO: mat multiplication with @ operator
#       mat @ vec with vec with 0 line? see pep


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
    
    def __init__(self, data, row, col, shape=None):
        
        #### validate the input ####

        # input lists same length?
        if not (len(data) == len(row) == len(col)):
            raise ValueError('data, row and col must have the same length!')

        # no negative indizes?
        if any([ x<0 or y<0 for x, y in zip(row, col)]):
            raise ValueError('indices must be positive!')
        
        # correct shape?
        if shape == None:
            self.shape = (max(row)+1, max(col)+1)
        else:
            assert len(shape) == 2
            assert type(shape[0]) == type(shape[1]) == int
            assert shape[0] > 0
            assert shape[1] > 0
            
            self.shape = shape

        self.content = dict()
        for d, r, c in zip(data, row, col):
            if d != 0:
                self.content[(r, c)] = d
    
    def __getitem__(self, index):
        self.__check_index(index)
        return self.content.get(index, 0)
    
    def __setitem__(self, index, value):
        self.__check_index(index)
        if value != 0:
            self.content[index] = value
        
    def __check_index(self, index):
        if type(index) != tuple or len(index) != 2:
            raise TypeError('Index must be a tuple like (x, y)')
        if type(index[0]) != int or type(index[1]) != int:
            raise TypeError('Index values must be of type int')
        if not ((0 <= index[0] <= self.shape[0]-1) and (0 <= index[1] <= self.shape[1]-1)):
            raise IndexError('Index out of matrix shape')
    

    def __repr__(self) -> str:
        return str(self.content)

    def __eq__(self, o: object) -> bool:
        return self.shape == o.shape and self.content == o.content

    def __add__(self, other):
        if self.shape != other.shape:
            raise ArithmeticError('Matrizes must have the same shape!')
        res = SparseMatrix([], [], [], shape=self.shape)
        content = {}
        for key in set(list(self.content.keys()) + list(other.content.keys())):
            content[key] = self.content.get(key, 0) + other.content.get(key, 0)
        res.content = content
        return res
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
            

    def __matmul__(self, other):
        res = SparseMatrix([], [], [], shape=(self.shape[0], other.shape[1]))
        
        combinations = [(i, k) for i in range(self.shape[0]) for k in range(other.shape[1])]
        
        for i, k in combinations:
            res[i, k] = sum(self[i, j] * other[j, k] for j in range(self.shape[1]))
        
        return res
    
    @property
    def T(self):
        shape = tuple(reversed(self.shape))
        content = dict()
        for key, value in self.content.items():
            content[tuple(reversed(key))] = value
        res = SparseMatrix([], [], [], shape=shape)
        res.content = content
        return res
