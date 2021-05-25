#!/usr/bin/env python3

import numpy as np
import scipy.sparse

# vergleichen Sie das Konvergenzverhalten grafisch


# --- linsolve.py --- #
"""
Hier Funktionen für iterative LGS Löser implementiert
"""

import numpy as np

# --- NYI ---


# --- sparse.py --- #
import numpy as np
import random

DIMENSIONS = 100

class SparseMatrix:
	"""
	Datenstruktur für (1), also schwach besetzte Matrizen

	@params:
			data - Wert des Elements
			row - Reihenindex des Elements
			col - Zeilenindex des Elements
	"""
	def __init__(self, data, row, col, m, n):
		self.data = data
		self.row = row
		self.col = col
		self.shape = (m, n)
		self.nel = len(self.data)
		idx = [(self.row[i], self.col[i]) for i in range(self.nel)]
		self.dict = {idx[i]: self.data[i] for i in range(self.nel)}
		
		
	
	def __repr__(self):
		return "SparseMatrix()"
	
	def __str__(self):
		return "(%s, %s) \t %s" % (self.row, self.col, self.data)
		# return "(" + self.row + ", " + self.col + ")" + "\t" + self.data
		
	# self[ij]
	def __getitem__(self, ij):
		if ij in self.dict.keys():
			return self.dict[ij[0], ij[1]]
		return 0.
		"""
		try:
			return self.dict[ij[0], ij[1]]
		except KeyError:
			return 0.
		"""
	
	# self[ij] = val
	def __setitem__(self, ij, val):
		if ij not in self.dict.keys():
			np.append(self.row, ij[0])
			np.append(self.col, ij[1])
			np.append(self.data, val)
			self.nel += 1
		else:
			idx = np.argwhere((self.row == ij[0]) & (self.col == ij[1]))
			self.data[idx] = val
		self.dict[ij] = val
		
		
	# bei self + other aufgerufen
	def __add__(self, other):
		# A einfach nehmen und alles aus B reinwerfen
		# C = A + B
		c = SparseMatrix(self.data, self.row, self.col, self.shape[0], self.shape[1])
		for other_entry in other.dict:
			c[other_entry] += other[other_entry]
					
		# ...
		return c
	# ...
	
	# elementweise mul bei bei self * other aufgerufen
	def __mul__(self, other):
		# forall entries in self check if there is an entry in other
		c = SparseMatrix(self.data, self.row, self.col, self.shape[0], self.shape[1])
		for c_entry in c.dict:
			# for every entry in self multiply it with the corresponding entry in other, if there is none its *0 because of getitem, if there is one, fine - we dont need to check the entries in other because 
			c[c_entry] *= other[c_entry]
			
		# ...
		return c
	# ...
	
	# bei self - other aufgerufen
	def __sub__(self, other):
		# A einfach nehmen und alles aus B reinwerfen
		# C = A - B
		c = SparseMatrix(self.data, self.row, self.col, self.shape[0], self.shape[1])
		for other_entry in other.dict:
			c[other_entry] -= other[other_entry]
			
		# ...
		return c
	# ...
	
	# matrizenmult bei self @ other aufgerufen
	def __matmul__(self, other):
		# ...
		pass
	# ...
		
	
# creates a sparse matrix with the given dimensions x and y and probability for non-zero entries z
def create_own_sparse(x, y, z):
	own_sparse = []
	for i in range(x):
		for j in range(y):
			n = random.uniform(0, 1)
			if n < z:
				data = random.uniform(0, 1)
				own_sparse.append(SparseMatrix(data, i, j))
	
	
	return own_sparse
# end of function

# checks if a sparse matrix A contains a non-zero entry at x , y
def own_sparse_is_nonz(A, x, y):
	
	if any( (entries.row == x and entries.col == y) for entries in A):
		return True
	
	return False
# end of function

# returns data at x y of A
def return_data(A, x, y):
	
	if own_sparse_is_nonz(A, x, y):
		for entries in A:
			if entries.row == x and entries.col == y:
				return entries.data
	
	# should never be reached if called correctly
	return null
# end of function

# adds all entries of two matrice A and B
def my_sparse_plus(A, B):
	own_sparse = []
	
	for i in range(DIMENSIONS):
		for j in range(DIMENSIONS):
			# both matrice have entries, actually add them
			if own_sparse_is_nonz(A, i, j) and own_sparse_is_nonz(B, i, j):
				own_sparse.append(SparseMatrix(return_data(A, i, j) + return_data(B, i, j), i, j))
			else:
				# only one has an entry, just take that
				if own_sparse_is_nonz(A, i, j):
					own_sparse.append(SparseMatrix(return_data(A, i, j), i, j))
				else:
					if own_sparse_is_nonz(B, i, j):
						own_sparse.append(SparseMatrix(return_data(B, i, j), i, j))
				
	# if no entry in both matrice it stays 0 aka not an entry
	
	return own_sparse
# end of function


# multiplies two matrice
def my_sparse_multi(A, B):
	own_sparse = []
	
	for i in range(DIMENSIONS):
		for j in range(DIMENSIONS):
			# both matrice have entries, actually add them
			if own_sparse_is_nonz(A, i, j) and own_sparse_is_nonz(B, i, j):
				own_sparse.append(SparseMatrix(return_data(A, i, j) * return_data(B, i, j), i, j))
						
	# if no entry in both matrice OR ONLY IN ONE! it stays 0 aka not an entry
						
	return own_sparse
# end of function

# gets two sparse matrice and links them 
def create_nonz_list(A, B):
	nonz_list = []
	
	for i in range(DIMENSIONS):
		for j in range(DIMENSIONS):
			if own_sparse_is_nonz(A, i, j) and own_sparse_is_nonz(B, i, j):
				# both lists have an entry at that point we need to merge somehow
				nonz_list.append(SparseMatrix(return_data(A, i, j), i, j))
				nonz_list.append(SparseMatrix(return_data(B, i, j), i, j))
			else:
				if own_sparse_is_nonz(A, i, j):
					nonz_list.append(SparseMatrix(return_data(A, i, j), i, j))
				else:
					if own_sparse_is_nonz(B, i, j):
						nonz_list.append(SparseMatrix(return_data(B, i, j), i, j))
				
	

	return nonz_list
# end of function
		
		

if __name__ == "__main__":
	row = np.genfromtxt("test_data/i.dat")
	col = np.genfromtxt("test_data/j.dat")
	data = np.genfromtxt("test_data/a_ij.dat")
	m = max(row)
	n = max(col)
	mat = SparseMatrix(data, row, col, m, n)                # Erzeugen einer Instanz durch Aufrufen des Konstruktors

	# getitem test
	print(mat[10,10])
	print(mat[10,100])

	# setitem test
	mat[10,100] = 1000.
	print(mat[10,100])

	# add test
	print(mat + mat)

	"""
	A = scipy.sparse.random(5, 5, 1)
	A_dense = A.toarray()
	own_sparse_A = create_own_sparse(DIMENSIONS,DIMENSIONS,0.1)
	own_sparse_B = create_own_sparse(DIMENSIONS,DIMENSIONS,0.1)
#	print(own_sparse_is_nonz(own_sparse_A, 0, 0))
#	for obj in own_sparse_A:
#		print(obj)
	
	own_sparse_C = my_sparse_plus(own_sparse_A, own_sparse_B)
	own_sparse_D = my_sparse_multi(own_sparse_A, own_sparse_B)
	
	print("NOW A")
	for i in own_sparse_A:
		print(i)
		
	print("NOW B")
	for i in own_sparse_B:
		print(i)
		
	print("NOW C")
	for i in own_sparse_C:
		print(i)
	
	
	print(A.nonzero())
	print(A_dense)
	print("now A")
	print(A)
	"""