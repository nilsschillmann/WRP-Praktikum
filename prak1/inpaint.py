from solve import solve_problem_jacobi
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from solve import solve_problem         # TODO Versteckte Funktion zur Demonstration



jacobi_iterations = 100
mg_iterations = 50



# Bild einlesen; ergibt
img = plt.imread("gauss.png")
#print(img.shape)

# Nur Rot-Wert verwenden (da die Werte bei Grautoenen gleich sind, koennte hier auch ein anderer Wert gebraucht werden)
img = img[:,:,0]
# Alternative
#img = img[...,0]

# Kopie des Bildes fuer spaetere Darstellung erzeugen
img_orig = img.copy()
img_jacobi = img.copy()
img_mg = img.copy()

# Array mit Wahrheitswerten (wahr wenn Pixel genau 1)
is_white = img == 1.

# Identifikation der zusammenhaengenden Regionen, wo is_white == True, als slices
regions = ndimage.find_objects(ndimage.label(is_white)[0])
#print(regions)
#print(len(regions))

# Ueber identifizierte Regionen iterieren
jacobi_residua = []
mg_residua = []

jacobi_results = []
mg_results = []


for region in regions:

    # Rand des Rechengebiets hinzufuegen
    domain = tuple([slice(r.start - 1, r.stop + 1) for r in region])
    
    # Problem durch Loesung der Laplace-Gleichung loesen; TODO: eigenen Code zur Loesung der Problems einfuegen
    mg_result, residua = solve_problem(img, domain, mg_iterations)
    mg_residua.append(residua[-1])
    mg_results.append(mg_result)
    img_mg[domain][1:-1, 1:-1] = mg_result
    
    jacobi_result, residua = solve_problem_jacobi(img, domain, jacobi_iterations)
    jacobi_residua.append(residua)
    jacobi_results.append(jacobi_result)
    img_jacobi[domain][1:-1, 1:-1] = jacobi_result


fig, axs = plt.subplots(1, 3)
axs[0].imshow(img_orig, cmap='gray')
axs[1].imshow(img_mg, cmap='gray')
axs[2].imshow(img_jacobi, cmap='gray')

axs[0].set_title("Original")
axs[1].set_title("Multigrid")
axs[2].set_title("Jacobi")

plt.show()


fig, axs = plt.subplots(len(regions), 3)
for ax, jc_result, mg_result, jc_residua, _mg_residua in zip(axs, jacobi_results, mg_results, jacobi_residua, mg_residua):
    ax[0].imshow(jc_result, cmap='gray', vmin=0, vmax=1)
    ax[1].imshow(mg_result, cmap='gray', vmin=0, vmax=1)
    ax[2].plot(jc_residua, c='red', label="jacobi")
    ax[2].plot(_mg_residua, c='blue', label="multigrid")
    ax[2].legend()

axs[0][0].set_title('Jacobi result')
axs[0][1].set_title('Multigrid result')
axs[0][2].set_title('Residuum after x-steps')

plt.show()






# fig, ax = plt.subplots()
# for jacobi_residua in jacobi_residua:
#     ax.plot(jacobi_residua, c='blue')

# for mg_residuum in mg_residua:
#     ax.plot(mg_residuum, c="red")

# plt.show()


# Originalbild und rekonstruiertes Bild darstellen und anzeigen


# fig = plt.figure()
# ax = [fig.add_subplot(121 + i) for i in range(2)]
# ax[0].imshow(img_orig, cmap='gray')
# ax[0].set_title("Orignal")
# ax[1].imshow(img, cmap='gray')
# ax[0].set_title("Reconstruction")
# plt.show()
