import numpy as np
import math
import matplotlib.pyplot as plt

# Fragen in der Übung:
# 1. wo kommt beim e das i her? welches sollen wir nehmen? (Formel fürs w)
# 2. was genau ist ein lowpass-filter? einfach die anteilig 90% größten Frequenzen rauslöschen oder verringern auf einen bestimmten Wert, welchen? oder was?
# 4. was ist ifft? wie soll das funktionieren / was soll es machen?

def fft(y, ifft=False):
	# vorwärtstransformation der fktwerte
	'''
	die fft funktioniert wie folgt:
		1. teile die N Datenwerte in zwei Hälften von 0 bis N/2 -1 und N/2 bis N-1
		Sei a = N/2
		2. für die erste Hälfte addiere den i-ten Eintrag mit dem i+a-ten Eintrag, also ersten Wert der ersten Hälfte mit erstem Wert der zweiten Hälfte etc
		3. für die untere Hälfte
			3.1 subtrahiere den j-a -ten Eintrag mit dem j-ten Eintrag, also ersten Wert der ersten Hälfte mit erstem Wert der zweiten Hälfte etc
			3.2 multipliziere mit w^i, wobei i dem j-a -ten Index entspricht und w = e^(-i*((2*PI)/N)
		4. wiederhole Schritt 1.-3. solange bis N = 2 ist
	Bei 2^P Werten brauchen wir P Schritte
    '''

	# c_k = 1/N sum_k=1^N-1 f_n 

	N = y.shape[0]

	if ifft:
		one_slash_N = 1
	else:
		one_slash_N = 1/N
	
	schritte = int(np.log2(N)) + 1

	y_copy = np.zeros(N, dtype=complex)
	c_k = np.zeros(N, dtype=complex)

	for i in range(N):
		y_copy[i] = y[i]

	y_copy_copy = y_copy.copy()

	bit_reversal = np.zeros(N)
	bit_reversal_level = 0
	
	# in der Darstellung gehen wir hier die x-Richtung durch
	for steps in range(1, schritte):
		
		alpha = int(N/2)
		index = 0
		
		steps = 2**(steps-1)
		# hier gehen wir in jedem Schritt die y-Richtung durch, also wie viele obere/untere Partitionen gibt es
		# dass wächst exponentiell und in jedem Schritt haben wir genau 2^(schritt-1) viele Partitionen
		for j in range(steps):

			# alpha beschreibt die größe der einzelnen Partitionen, also wie viele Elemente müssen in jedem Schritt
			# in jede Hälfte, also 4, 2, 1 bei N=8
			# der index zaehlt dann die tatsächlichen indize der Fi hoch
			
			laufvar = 0
			# obere haelfte
			while laufvar < alpha:
				y_copy_copy[index] = y_copy[index] + y_copy[index+alpha]
				index+=1
				laufvar+=1
			
			laufvar = 0
			# untere hälfte
			while laufvar < alpha:
				minus = y_copy[index-alpha] - y_copy[index]
				if (index-alpha)==0:
					w = 1
				else:
					# w = e^(-i*((2*PI)/N)
					w = np.exp((1j)*((2*np.pi)/N))

				if ifft:
					y_copy_copy[index] = w**(index-alpha) * minus
				else:
					y_copy_copy[index] = w**((-1)*(index-alpha)) * minus
				bit_reversal[index] += 2**bit_reversal_level
				# da nur im - Zweig das bit-reversal relevant ist nur hier, im letzten Schritt hat das Bit die höchste Wertigkeit, starten also bei
				# bit_reversal_level=0 für den ersten Schritt, wenn wir da subtrahieren also +1, im zweiten Schritt +2 etc
				index+=1
				laufvar+=1
			
		N = alpha
		y_copy[:] = y_copy_copy
		bit_reversal_level += 1
	

	for i in range(c_k.shape[0]):
		bit_index = int(bit_reversal[i])
		# print("c_k[",bit_index,"] =",i)
		c_k[bit_index] = one_slash_N * y_copy[i]
		# ordnen jetzt jedem indize aus bit_reversal den korrekten Wert zu
	
	return c_k
	
def low_pass(y):
	N = y.shape[0]
	ten_percent_half = int(np.floor((0.1*N)/2))
	low_passed = np.zeros(N, dtype=complex)
	for i in range(ten_percent_half):
		low_passed[i] = y[i]
		low_passed[-i] = y[-i]
	
	return low_passed

if __name__ == '__main__':
	test = False
	if test:
		N = 2**10
		# Erzeugen zufälliger Amplituden
		rnd = np.random.RandomState(12345)
		c_real = rnd.uniform(low=-1, high=1, size=N)
		c_imag = rnd.uniform(low=-1, high=1, size=N)
		c_ref = c_real + c_imag*1j
		# aus Amplituden die fktwerte erzeugen
		f_ref = np.zeros(N, dtype=complex)
		for n in range(N):
			for k in range(N):
				f_ref[n] += c_ref[k]*np.exp(1j*2*np.pi/N*k*n)
		# Faktor 1/N nur in eine Richtung verwenden
		c = fft(f_ref)
		assert np.allclose(c, c_ref)
		f = fft(c, ifft=True)
		assert np.allclose(f, f_ref)
	
	# links t, rechts f(t)
	y = np.genfromtxt("signal.dat")
	data_points = y.shape[0]
	y_t = np.zeros(data_points)
	y_f_t = np.zeros(data_points)
	for i in range(data_points):
		y_t[i] = y[i][0]
		y_f_t[i] = y[i][1]
	# y = np.array([0,1,2,3,4,5,6,7])
	fft_of_y_f_t = fft(y_f_t)
	low_passed_y = low_pass(fft_of_y_f_t)
	ifft_of_y_f_t = fft(low_passed_y, ifft=True)

	if False:
		fig, (ax1, ax2) = plt.subplots(2)
		ax1.set_title("--- given ---")
		ax1.plot(y_t, y_f_t)
		ax2.set_title("\n fft -> lowpass -> ifft")
		ax2.plot(y_t, ifft_of_y_f_t)
		plt.show()
	else:
		plt.plot(y_t, y_f_t)
		plt.plot(y_t, ifft_of_y_f_t, alpha=0.75)
		plt.legend(["given", "fft -> lowpass -> ifft"])
		plt.show()

	'''
	fig, (ax1, ax2, ax3) = plt.subplots(3)
	# fig.suptitle('Given (top) and calculated via fft (bottom) values of f(t)')
	ax1.set_title("Given values f(t)")
	ax1.plot(y_t, y_f_t)
	ax2.set_title("Calculated values f(t)")
	ax2.plot(y_t, fft_of_y_t)
	ax3.set_title("Low pass values of calculated values")
	ax3.plot(y_t, low_passed_y)
	plt.show()
	'''
	'''
	plt.plot(y_t, y_f_t)
	plt.plot(y_t, fft_of_y_t)
	plt.plot(y_t, low_passed_y)
	plt.legend(["Reference", "FFT", "LOWPASS"])
	plt.show()
	'''
	