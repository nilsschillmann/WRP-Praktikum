import numpy as np
from numpy.lib.index_tricks import fill_diagonal
import scipy as sp
from linsolve import jacobi_method


# TODO. Gleichungssystem aufbauen. Beispiel in der Übung

def solve_problem(img_in, domain_in):
    img = img_in.copy()
    domain = domain_in.copy()

    def gpp(x):
        return - 12. * x**2 

    restrict = inject #statt weight?
    n_min = min(img[domain].shape)
    nLevels = int(np.log2(n_min - 1) + 0.1)
    nFineGrid = np.array(img[domain].shape, dtype=int) - 1
    
    n = np.zeros(nLevels,dtype='int,int')
    n[0] = nFineGrid
    for i in range(nLevels):
        x = nFineGrid[0] // 2**i
        y = nFineGrid[1] // 2**i
        n[i] = (x, y)

    x = []       # grid hierarchy
    u = []       # solution, level 0: u, level >= 1: error  
    f = []       # RHS, f (A u = f) or residuum r = f - A u
    r = []       # residuum r = f - A u
    h = []       # grid spacing

    for i in range(nLevels):
        x.append(np.linspace(-1, 1, n[i] + 1))    # grids
        u.append(np.zeros(n[i]+1))
        f.append(gpp(x[i]))               # residuum vector
        r.append(np.zeros(n[i]+1))               # residuum vector
        h.append(x[i][1] - x[i][0])

    u[0][:]  = img[domain]

    doFullMGVCycle(nLevels, u, f, r, h, nu1=1, nu2=1, omega=2./3.)

    return img[domain]


def inject(u, u2):
    """simple copy from fine to coarse grid"""
    u2[:, :] = u[::2, ::2]
    return u2

def prolong(u, u2):
    """linear interpolation u2 -> u """
    #u[0::2, 0::2] = u2[0::, 0::]
    #u[::2,1::2] = ...
    # 
    u[1::2, 1::2] = 0.5 * (u2[0:-1:, 0:-1:] + u2[1::, 1::])
    # vertikaler Sweep
    # averages of 4
    return u

def prolongAndCorrect(u, u2):
    """linear interpolation u2 -> u """
    u[0::2, 0::2] = u[0::2, 0::2] + u2[0::, 0::]
    u[1::2, 1::2] = u[1::2, 1::2] + 0.5 * (u2[0:-1:, 0:-1:] + u2[1::, 1::])
    return u

def doVCycle(level, nLevels, u, f, r, h, nu1=3, nu2=1, omega=1.):
    """performs a complete V-cycle starting at grid level 'level' down to the
    finest level 
    Note: level here is the index in the hierarchy with value 0 for
    the finest grid and value (nLevels-1) for the coarsest grid 
    """
    for l in range(level, nLevels-1):
        for _ in range(nu1):
            stepJacobiTriDiag(u[l], f[l], h[l], omega)
            """
                    [-4 1 ... 1 ...
            A =       1 -4 1 ... 1 ... 
                     ...................
            b ergibt sich aus den Randwerten (und bei Poisson-Gleichung aus Funktionswerten)
            """
            A, b = assemble_system(u[l])
            u_flat, r_flat = jacobi_method(A, b, 1, u[l][1:-1,1:-1].flatten(), w=1, threshold=1e-100, full_output=True)
        residuum(r[l], u[l], f[l], h[l])
        restrict(r[l], f[l+1])  # RHS for next level
        u[l+1][:] = 0.0         # initial condition for next level (error eq.) 
    
    for i in range(nu2):
        stepJacobiTriDiag(u[-1], f[-1], h[-1], omega)
      
    for l in range(nLevels-1, level, -1):
        prolongAndCorrect(u[l-1], u[l])
        for i in range(nu2):
            stepJacobiTriDiag(u[l-1], f[l-1], h[l-1], omega)


def assemble_system(u):
    u_flat = u.flatten()
    A = build_A(u)
    b = np.zeros(shape=(A.shape[0]))
    return A, b
    
    
def build_A(u_mit_rand):
    u = u_mit_rand[1:-1:, 1:-1:]
    A = np.zeros(shape=(u.size, u.size))
    np.fill_diagonal(A, 4)
    b = np.zeros(shape=(u.size))
    for x, y in [(x, y) for x in range(u.shape[0]) for y in range(u.shape[1])]:
        indices = []
        if x-1 >= 0:
            indices.append(np.ravel_multi_index(([x-1], [y]), dims=u.shape)[0])
        if x+1 < u.shape[0]:
            indices.append(np.ravel_multi_index(([x+1], [y]), dims=u.shape)[0])
        if y-1 >= 0:
            indices.append(np.ravel_multi_index(([x], [y-1]), dims=u.shape)[0])
        if y+1 < u.shape[1]:
            indices.append(np.ravel_multi_index(([x], [y+1]), dims=u.shape)[0])

        x_index = np.ravel_multi_index(([x], [y]), dims=u.shape)[0]
        for i in indices:
            A[x_index, i] = -1
            
    
    
    
    for x, y in [(x, y) for x in range(u_mit_rand.shape[0]) for y in range(u_mit_rand.shape[1])]:
        if x-1 == 0:
            # Pixel am linken Rand
            # i vom Pixel x,y und Wert am Rand x-1,y
            b[] += u[x-1][y]
            pass
        if x+1 == u.shape[0]-1:
            # Pixel am rechten Rand
            # b[i] += (Wert an Stelle x+1, y)
            pass
        if y-1 == 0:
            pass
        if y+1 == u.shape[1]-1:
            pass


    return A

def doFullMGVCycle(nLevels, u, f, r, h, nu1=3, nu2=1, omega=1.):
    
    for l in range(nLevels-1):
        inject(f[l], f[l+1])
    # solve on coarsest grid
    for i in range(nu2):
        stepJacobiTriDiag(u[-1], f[-1], h[-1], omega)
    for l in range(nLevels-1, 0, -1):
        prolong(u[l-1], u[l])
        doVCycle(l-1, nLevels, u, f, r, h, nu1=3, nu2=3, omega=2./3.)
        
if __name__ == "__main__":
    u = np.arange(9).reshape((3, 3))
    A = build_A(u)
    print(u)
    print(A)