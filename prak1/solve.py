import numpy as np
from numpy.lib.index_tricks import fill_diagonal
import scipy as sp
from linsolve import jacobi_method


# TODO. Gleichungssystem aufbauen. Beispiel in der Übung
# v zyklus implementieren

'''
HIER CODE
def v_zyk(v,F,A):
    #A,b = create_matrix(v,F)
    b = F
    iter = np.linalg.solve(A,b)#Iteration(A,F,x = v)
    if not grid_test():
        new_F = restrict_vector(F-A@iter,1,A)
        y = np.zeros(new_F.shape)
        y = v_zyk(y,new_F)
        result = iter+prolong(y,1,A)
    corr = np.linalg.solve(A,F)#Iteration(A,F, x=result)
    return result

def full_multigrid(v,F,A):
    if grid_test(A.shape[0],A.shape[1]):
        v[:] = 0
    else:
        new_F = restrict_vector(F.flatten(),1,A.shape)
        y = np.zeros(new_F.flatten().shape) 
        y = full_multigrid(y,new_F,new_F)
        corr = prolong(y,1,A)
    result = v_zyk(v,F)
    return result
'''
def stepJacobiTriDiag(u, f, h, omega=1):
    """Performs one step of the damped Jacobian iteration for the linear system
    A u = f, where A is the discrete Laplacian u_{i-1} - 2 u_i + u_{i+1}

    Parameters
    ----------
    u : iterated solution vector, u[0] and u[-1] hold the Dirichlet b.c. values
    f : right hand side of the system A_h u = - f 

    Return
    ------
    u : the new iterated value
    """
    h2 = h * h
    n = u.shape[0]
    u[1:n-1] = (1. - omega) * u[1:n-1] \
            + omega * 0.5 * (-h2 * f[1:n-1] + u[2:n] + u[0:n-2])
    return u



def solve_problem_jacobi(img, domain):
    patch = img.copy()[domain]
    A, b = assemble_system(patch)
    u_flat = jacobi_method(A, b, 100, patch[1:-1, 1:-1].flatten(), w=1, threshold=0, full_output=False)
    return u_flat.reshape((patch.shape[0]-2, patch.shape[1]-2))
    

def solve_problem(img, domain):
    patch = img.copy()[domain]
    result = multigrid(patch)
    return result[1:-1, 1:-1]

def multigrid(img):
    grids = build_grids(img)
    #starte mit gröbstem gitter
    jacobi_step(grids[0])
    #für jedes gitter
    #   prolonguiere
    #   vzyklus
    for i in range(len(grids)-1):
        new_grid = prolong(grids[i+1].copy(), grids[i])
        grids[i+1][1:-1, 1:-1] = new_grid[1:-1, 1:-1]
        v_cycle(grids, i)
    
    return grids[-1]
    
def v_cycle(grids, level):
    #go down
    for i in range(level, -1, -1):
        grids[i] = inject(grids[i+1], grids[i].copy())
        jacobi_step(grids[i])
    
    #go up
    for i in range(level):
        new_grid = prolong(grids[i+1].copy(), grids[i])
        grids[i+1][1:-1, 1:-1] = new_grid[1:-1, 1:-1]
        jacobi_step(grids[i+1])

def jacobi_step(img):
    A, b = assemble_system(img)
    result, residuum = jacobi_method(A, b,iterations=1,threshold=0,full_output=True)
    img[1:-1, 1:-1] = result.reshape((img[1:-1, 1:-1].shape))

def build_grids(img):
    nLevels = int(np.log2(min(img.shape) - 1) + 0.1)
    grids = []
    for i in range(nLevels):
        steps = min(img.shape) // 2**(i+1)
        grids.append(img[::steps, ::steps])
    return grids
    
    
    

def solve_problem_old(img_in, domain_in):
    img = img_in.copy()
    domain = domain_in

    def gpp(x):
        return - 12. * x**2 

    restrict = inject #statt weight?
    n_min = min(img[domain].shape)
    nLevels = int(np.log2(n_min - 1) + 0.1)
    nFineGrid = np.array(img[domain].shape, dtype=int) - 1
    
    n = np.zeros(nLevels, dtype='int,int')
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


def doFullMGVCycle(nLevels, u, f, r, h, nu1=3, nu2=1, omega=1.):
    
    for l in range(nLevels-1):
        inject(f[l], f[l+1])
    # solve on coarsest grid
    for i in range(nu2):
        A, b = assemble_system(u[-1])
        u_flat, r_flat = jacobi_method(A, b, 
                                        iterations=1, 
                                        startvector=u[-1][1:-1,1:-1].flatten(),
                                        w=omega, 
                                        threshold=0,
                                        full_output=True)
        
        stepJacobiTriDiag(u[-1], f[-1], h[-1], omega)
    for l in range(nLevels-1, 0, -1):
        prolong(u[l-1], u[l])
        doVCycle(l-1, nLevels, u, f, r, h, nu1=3, nu2=3, omega=2./3.)


def doVCycle(level, nLevels, u, f, r, h, nu1=3, nu2=1, omega=1.):
    """performs a complete V-cycle starting at grid level 'level' down to the
    finest level 
    Note: level here is the index in the hierarchy with value 0 for
    the finest grid and value (nLevels-1) for the coarsest grid 
    """
    for l in range(level, nLevels-1):
        for _ in range(nu1):
            stepJacobiTriDiag(u[l], f[l], h[l], omega)
            A, b = assemble_system(u[l])
            u_flat, r_flat = jacobi_method(A, b, iterations=1, 
                                           startvector = u[l][1:-1,1:-1].flatten(), 
                                           w=1, 
                                           threshold=1e-100, 
                                           full_output=True)
        residuum(r[l], u[l], f[l], h[l])
        restrict(r[l], f[l+1])  # RHS for next level
        u[l+1][:] = 0.0         # initial condition for next level (error eq.) 
    
    for i in range(nu2):
        stepJacobiTriDiag(u[-1], f[-1], h[-1], omega)
      
    for l in range(nLevels-1, level, -1):
        prolongAndCorrect(u[l-1], u[l])
        for i in range(nu2):
            stepJacobiTriDiag(u[l-1], f[l-1], h[l-1], omega)


def inject(u, u2):
    """simple copy from fine to coarse grid"""
    u2[:, :] = u[::2, ::2]
    return u2

def prolong(u, u2):
    # horizontal sweep
    u[::2, ::2] = u2[::, ::]
    u[::2, 1::2] = 0.5 * (u2[0::, 0:-1:] + u2[::, 1::])
    # vertikaler Sweep
    u[1::2, ::2] = 0.5 * (u2[0:-1:, 0::] + u2[1::, ::])
    # averages of 4 = 1/4 * (v_i,j + v_i+1,j + v_i,j+1, v_i+1,j+1)
    #u[1::2, 1::2] = 0.25 * (u2[0:-1:, 0:-1:] + u2[1::, 0:-1] + u2[0:-1, 1::] + u2[1::, 1::]
    u[1::2, 1::2] = 0.25 * (u2[0:-1:, 0:-1:] + u2[1::, 1::] + u2[1::, 0:-1] +u2[0:-1:, 1:])
    return u

def prolongAndCorrect(u, u2):
    """linear interpolation u2 -> u """
    u[0::2, 0::2] = u[0::2, 0::2] + u2[0::, 0::]
    u[1::2, 1::2] = u[1::2, 1::2] + 0.5 * (u2[0:-1:, 0:-1:] + u2[1::, 1::])
    return u

    
    
def assemble_system(u_mit_rand):
    u = u_mit_rand[1:-1:, 1:-1:]
    A = np.zeros(shape=(u.size, u.size))
    np.fill_diagonal(A, 4)
    b = np.zeros(shape=(u.size))
    for x, y in [(x, y) for x in range(u.shape[0]) for y in range(u.shape[1])]:
        #print(x, y)
        y_indices = []
        if x-1 >= 0:
            y_indices.append(np.ravel_multi_index(([x-1], [y]), dims=u.shape)[0])
        if x+1 < u.shape[0]:
            y_indices.append(np.ravel_multi_index(([x+1], [y]), dims=u.shape)[0])
        if y-1 >= 0:
            y_indices.append(np.ravel_multi_index(([x], [y-1]), dims=u.shape)[0])
        if y+1 < u.shape[1]:
            y_indices.append(np.ravel_multi_index(([x], [y+1]), dims=u.shape)[0])

        x_index = np.ravel_multi_index(([x], [y]), dims=u.shape)[0]
        for i in y_indices:
            A[x_index, i] = -1
    
    for x, y in [(x, y) for x in range(1, u_mit_rand.shape[0]-1) for y in range(1, u_mit_rand.shape[1]-1)]:
        #print(x, y)
        border_values = []
        if x-1 == 0:
            # Pixel am linken Rand
            # i vom Pixel x,y und Wert am Rand x-1,y
            border_values.append(u_mit_rand[x-1, y])
        if x+1 == u_mit_rand.shape[0]-1:
            # Pixel am rechten Rand
            # b[i] += (Wert an Stelle x+1, y)
            border_values.append(u_mit_rand[x+1, y])
        if y-1 == 0:
            border_values.append(u_mit_rand[x, y-1])
        if y+1 == u_mit_rand.shape[1]-1:
            border_values.append(u_mit_rand[x, y+1])
        b_value = sum(border_values, start=0)
        b_index = np.ravel_multi_index(([x-1], [y-1]), dims=u.shape)[0]
        b[b_index] = b_value


    return A, b

        
if __name__ == "__main__":
    from linsolve import jacobi_method
    u = np.arange(25).reshape((5, 5))
    A, b = assemble_system(u)
    print(u)
    print(b)
    print(A)
    
    u_flat, r_flat = jacobi_method(A, b, 1, u[1:-1,1:-1].flatten(), w=1, threshold=1e-100, full_output=True)
    
    print(u_flat)
    print(r_flat)