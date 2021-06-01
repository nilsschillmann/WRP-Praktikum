"""
Hier sollen die Funktionen für iterative LGS-Löser implementiert werden.
"""

# TODO: konvergenzkriterium einbauen (vom residuum abhängig machen)
# TODO: zwischenergebnisse von außen zugänglich machen (full output mit dictionary)
# TODO: (funktionen numpy kompatibel machen)
# TODO: gaus_seidel und SOR auf korrektheit überprüfen

import numpy as np
from sparse import SparseMatrix

def jacobi_method(A, b, iterations=5, startvector=None, w=1):
    
    Dinvers_content = {key: 1/A[key] for key in A.content.keys() if key[0] == key[1]}
    D_invers = SparseMatrix([], [], [], shape=A.shape)
    D_invers.content = Dinvers_content
    Diw = D_invers * w
    
    x_old = startvector or SparseMatrix([], [], [], shape=b.shape)
      
    for k in range(iterations):
        # Vorlesung 21.04 S. 4
        x_new = x_old + Diw @ (b - (A @ x_old))
        x_old = x_new

    return x_new


def gaus_seidel(A, b, iterations=5, startvector=None):
    return SOR(A, b, iterations, startvector, w=1)       
    

def SOR(A, b, iterations=5, startvector=None, w=1/2):
    
    D = {key: A[key] for key in A.content.keys() if key[0] == key[1]}
    R = {key: A[key] for key in A.content.keys() if key[0] < key[1]}
    L = {key: A[key] for key in A.content.keys() if key[0] > key[1]}
    I = {key: 1 for key in A.content.keys() if key[0] == key[1]}

    # for i in 1 ... laenge x
    # berechne si
    # berechne xi
    # Gauß-Seidel: -> berechne die si mit den geupdateten xi (in jacobi würde man ERST ALLE si berechnen)
    # für k+1: s_i = A_zeile_i * x^k
    # für k+1 für die i-te komponente: x^k+1_i = x^k_i - w/(A_i,i) * (s_i - b_i)
    # Jacobi si, si, si, ..., xi, xi, xi, ...
    # Gauß-Seidel si, xi, si, xi, ...
    # SOR -> 1/(A-i,i) wird zu w/(A_i,i)
    
    
    # aus dem video
    x_old = startvector or SparseMatrix([], [], [], shape=b.shape)
    x_new = startvector or SparseMatrix([], [], [], shape=b.shape)
    for k in range(iterations):
        for i in range(b.shape[0]):
            
            #A_i_content = {key: A[key] for key in A.content.keys() if key[0] == i}
            A_i_content = {(0, key[1]): A[key] for key in A.content.keys() if key[0] == i}# hier war der fehler
            A_i = SparseMatrix([], [], [], shape=(1, A.shape[1]))
            A_i.content = A_i_content
#            print("Ai ", A_i)
            
            s_i = (A_i @ x_old)
            wert = s_i[0, 0]
            #print(A_i, " @ ", x_old, "=", s_i)
            #print("s", i, " = ", wert)
            x_new[i, 0] = x_old[i, 0] - w/(A[i,i]) * (wert - b[i, 0])
            x_old = 1 * x_new # TODO: copy methode implementieren
        # eine Iteration, ein mal den Lösungsvektor
        # if ...:
        #     break
    # k Iterations

    return x_old

if __name__ == '__main__':
    
    ## example from the english wikipedia
    # https://en.wikipedia.org/wiki/Jacobi_method
#    A = SparseMatrix([2, 1, 5, 7], [0, 0, 1, 1], [0, 1, 0, 1])
#    b = SparseMatrix([11, 13], [0, 1], [0, 0])
#    x0 = SparseMatrix([1, 1], [0, 1], [0, 0])
    
    A = SparseMatrix([4, -2, 1, -1, 5, -2, 1, 1, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2])
    b = SparseMatrix([2, 4, 6], [0, 1, 2], [0, 0, 0])
    x0 = SparseMatrix([1, 2, 1], [0, 1, 2], [0, 0, 0])
    
    x_jac1 = SparseMatrix([1.25, 1.4, 1], [0, 1, 2], [0, 0, 0])
    x_gasei1 = SparseMatrix([1.25, 1.45, 1.1], [0, 1, 2], [0, 0, 0])
    
    print("--- JACOBI ---")
    res = jacobi_method(A, b, iterations=1, startvector=x0, w=1)
    print("1 iterations: ", res)
    print("EXPECTED: 1 iterations: ", x_jac1)
    
    res = jacobi_method(A, b, iterations=2, startvector=x0, w=1)
    print("2 iterations: ", res)
    
    res = jacobi_method(A, b, iterations=25, startvector=x0, w=1)
    print("25 iterations: ", res)
    
    print("--- GAUß-SEIDEL ---")
    res = gaus_seidel(A, b, iterations=1, startvector=x0)
    print("1 iterations: ", res)
    print("EXPECTED: 1 iterations: ", x_gasei1)
    
    res = gaus_seidel(A, b, iterations=2, startvector=x0)
    print("2 iterations: ", res)
    
    res = SOR(A, b, iterations=25, startvector=x0)
    print("25 iterations: ", res)
    