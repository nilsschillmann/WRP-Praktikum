#%% imports
from sparse import SparseMatrix
from linsolve import jacobi_method
from linsolve import SOR
from linsolve import gaus_seidel

import numpy as np
from matplotlib import pyplot as plt


#%% load data

# a_ij = (np.loadtxt("./test_data/a_ij.dat")).tolist()
# i_index = (np.loadtxt("./test_data/i.dat", dtype=int)).tolist()
# j_index = (np.loadtxt("./test_data/j.dat", dtype=int)).tolist()
# b_vector = (np.loadtxt("./test_data/b.dat")).tolist()


# %% generate sparse matrizes

# A = SparseMatrix(a_ij, i_index, j_index)
# b = SparseMatrix(b_vector, list(range(A.shape[0])), [0 for i in range(A.shape[1])])

#%%
A = SparseMatrix([4, -2, 1, -1, 5, -2, 1, 1, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2])
b = SparseMatrix([2, 4, 6], [0, 1, 2], [0, 0, 0])
#x0 = SparseMatrix([1, 2, 1], [0, 1, 2], [0, 0, 0])
# %% run linsolve Methods

th = 0.0001
max_its = 10

print('jacobi')
jacobi_res, jacobi_resids = jacobi_method(A, b, threshold=th, iterations=max_its, full_output=True)

print('gaus seidel')
gs_res, gs_resids = gaus_seidel(A, b, threshold= th, iterations=max_its, full_output=True)

print('SOR')
sor_res, sor_resids = SOR(A, b, threshold= th, iterations=max_its, full_output=True)

print('end')
# %% plot results

fig, ax = plt.subplots(dpi=200)
ax.plot(jacobi_resids, label='jacobi')
ax.plot(gs_resids, label='gauß seidel')
ax.plot(sor_resids, label='sor')

ax.legend()

ax.set_ylabel('max( residuum )')
ax.set_xlabel('iterations')




# %%
