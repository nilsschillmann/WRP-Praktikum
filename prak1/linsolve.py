"""
Hier sollen die Funktionen für iterative LGS-Löser implementiert werden.
"""

import numpy as np
from sparse import SparseMatrix

def jacobi_method(A, b, iterations=5, startvector=None, w=1):
    
    Dinvers_content = {key: 1/A[key] for key in A.content.keys() if key[0] == key[1]}
    D_invers = SparseMatrix([], [], [], shape=A.shape)
    D_invers.content = Dinvers_content
    Diw = D_invers * w
    
    if startvector is None:
        startvector = SparseMatrix([], [], [], shape=(b.shape[0], 1))
    x_old = startvector
      
    for k in range(iterations):
        x_new = x_old + Diw @ (b - (A @ x_old))
        x_old = x_new

    return x_new

def gaus_seidel(A, b):
    pass

def SOR(A, b):
    pass

if __name__ == '__main__':
    
    ## example from the english wikipedia
    # https://en.wikipedia.org/wiki/Jacobi_method
    A = SparseMatrix([2, 1, 5, 7], [0, 0, 1, 1], [0, 1, 0, 1])
    b = SparseMatrix([11, 13], [0, 1], [0, 0])
    x0 = SparseMatrix([1, 1], [0, 1], [0, 0])
        
    x1 = SparseMatrix([1.25, 1.4, 1], [0, 1, 2], [0, 0, 0])
    res = jacobi_method(A, b, iterations=1, startvector=x0, w=1)
    print("1 iterations: ", res)
    
    res = jacobi_method(A, b, iterations=2, startvector=x0, w=1)
    print("2 iterations: ", res)
    
    res = jacobi_method(A, b, iterations=25, startvector=x0, w=1)
    print("25 iterations: ", res)
    