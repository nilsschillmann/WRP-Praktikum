"""
Hier sollen die Funktionen für iterative LGS-Löser implementiert werden.
"""

import numpy as np
from sparse import SparseMatrix

def jacobi_method(A, b, iterations=5, startvector=None, w=1):
    
    Dinvers_content = {key: w/A[key] for key in A.content.keys() if key[0] == key[1]}
    D_invers = SparseMatrix([], [], [], shape=A.shape)
    D_invers.content = Dinvers_content
    
    if startvector is None:
        startvector = SparseMatrix([], [], [], shape=(b.shape[0], 1))
    
    #k_plus1vektor = np.zeros(len(b))
    x_old = startvector
      
    for k in range(iterations):
        #print("Jacobi: ", k)
        x_new = x_old + D_invers @ (b - (A @ x_old))
        x_old = x_new

    return x_new

def gaus_seidel(A, b):
    pass

def SOR(A, b):
    pass

if __name__ == '__main__':
    # row = list(np.genfromtxt("test_data/i.dat", dtype=int))
    # col = list(np.genfromtxt("test_data/j.dat", dtype=int))
    # data = list(np.genfromtxt("test_data/a_ij.dat"))
    # mat = SparseMatrix(data, row, col)                # Erzeugen einer Instanz durch Aufrufen des Konstruktors
    
    # b_data = list(np.genfromtxt("test_data/b.dat"))
    # b_mat = SparseMatrix(b_data, list(np.arange(len(b_data))), list(np.zeros(len(b_data)) ))

    # # getitem test
    # # print(mat[10,10])
    # # print(mat[10,100])

    # # # setitem test
    # # mat[10,100] = 1000.
    # # print(mat[10,100])
    
    # print(type(col[12]))
    # print(type(row[12]))
    # print('mat shape', mat.shape)

    # # add test
    # #print(mat + mat)
    
    # jacobi_method(mat, b_mat)
    
    # A = SparseMatrix([4, -2, 1, -1, 5, -2, 1, 1, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2])
    # b = SparseMatrix([2, 4, 6], [0, 1, 2], [0, 0, 0])
    # x0 = SparseMatrix([1, 2, 1], [0, 1, 2], [0, 0, 0])
        
    # x1 = SparseMatrix([1.25, 1.4, 1], [0, 1, 2], [0, 0, 0])
    # res = jacobi_method(A, b, iterations=1, startvector=x0, w=1)
    # print(res)
    
    
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
    