import numpy as np
from linsolve import jacobi_method

def solve_problem_jacobi(img, domain, iterations):
    patch = img.copy()[domain]
    A, b = assemble_system(patch)
    u_flat, residua = jacobi_method(A, b, iterations=iterations, startvector=patch[1:-1, 1:-1].flatten(), w=1, threshold=0, full_output=True)
    return u_flat.reshape((patch.shape[0]-2, patch.shape[1]-2)), residua
    

def solve_problem(img, domain):
    patch = img.copy()[domain]
    result, residua = multigrid(patch)
    return result[1:-1, 1:-1], residua

def multigrid(img):
    residua = []
    grids = build_grids(img)
    #starte mit gröbstem gitter
    jacobi_step(grids[0], residua)
    #für jedes gitter
    #   prolonguiere
    #   vzyklus
    for i in range(len(grids)-1):
        new_grid = prolong(grids[i+1].copy(), grids[i])
        grids[i+1][1:-1, 1:-1] = new_grid[1:-1, 1:-1]
        v_cycle(grids, i, residua)
    
    return grids[-1], residua
    
def v_cycle(grids, level, residua):
    #go down
    
    for i in range(level, -1, -1):
        grids[i] = inject(grids[i+1], grids[i].copy())
        jacobi_step(grids[i], residua)
    
    #go up
    for i in range(level):
        new_grid = prolong(grids[i+1].copy(), grids[i])
        grids[i+1][1:-1, 1:-1] = new_grid[1:-1, 1:-1]
        jacobi_step(grids[i+1], residua)

def jacobi_step(img, residua = None):
    A, b = assemble_system(img)
    result, residuum = jacobi_method(A, b,iterations=20,threshold=0,full_output=True)
    img[1:-1, 1:-1] = result.reshape((img[1:-1, 1:-1].shape))
    if residua is not None:
        #print(residuum)
        residua.append(residuum)

def build_grids(img):
    nLevels = int(np.log2(min(img.shape) - 1) + 0.1)
    grids = []
    for i in range(nLevels):
        steps = min(img.shape) // 2**(i+1)
        grids.append(img[::steps, ::steps])
    return grids
    
    
   
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
