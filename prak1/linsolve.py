"""
Hier sollen die Funktionen für iterative LGS-Löser implementiert werden.
"""

import numpy as np
from sparse import SparseMatrix

def jacobi_method(A, b, iterations=5, starvector=None, w=1):
    
    Dinvers_content = {key: w/A[key] for key in A.content.keys() if key[0] == key[1]}
    D_invers = SparseMatrix([], [], [], shape=A.shape)
    D_invers.content = Dinvers_content
    
    if starvector == None:
        starvector = SparseMatrix([], [], [], shape=(b.shape[0], 1))
    
    #k_plus1vektor = np.zeros(len(b))
    x_old = starvector
      
    for k in range(iterations):
        print("Jacobi: ", k)
        x_new = x_old + D_invers @ (b - (A @ x_old))
        x_old = x_new

    print(x_new)

def gaus_seidel(A, b):
    pass

def SOR(A, b):
    pass

if __name__ == '__main__':
    row = list(np.genfromtxt("test_data/i.dat", dtype=int))
    col = list(np.genfromtxt("test_data/j.dat", dtype=int))
    data = list(np.genfromtxt("test_data/a_ij.dat"))
    mat = SparseMatrix(data, row, col)                # Erzeugen einer Instanz durch Aufrufen des Konstruktors
    
    b_data = list(np.genfromtxt("test_data/b.dat"))
    b_mat = SparseMatrix(b_data, list(np.arange(len(b_data))), list(np.zeros(len(b_data)) ))

    # getitem test
    # print(mat[10,10])
    # print(mat[10,100])

    # # setitem test
    # mat[10,100] = 1000.
    # print(mat[10,100])
    
    print(type(col[12]))
    print(type(row[12]))
    print('mat shape', mat.shape)

    # add test
    #print(mat + mat)
    
    jacobi_method(mat, b_mat)
    