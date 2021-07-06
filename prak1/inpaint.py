import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
#from solve import solve_problem         # TODO Versteckte Funktion zur Demonstration

# Bild einlesen; ergibt
img = plt.imread("gauss.png")
#print(img.shape)

# Nur Rot-Wert verwenden (da die Werte bei Grautoenen gleich sind, koennte hier auch ein anderer Wert gebraucht werden)
img = img[:,:,0]
# Alternative
#img = img[...,0]

# Kopie des Bildes fuer spaetere Darstellung erzeugen
img_orig = img.copy()

# Array mit Wahrheitswerten (wahr wenn Pixel genau 1)
is_white = img == 1.

# Identifikation der zusammenhaengenden Regionen, wo is_white == True, als slices
regions = ndimage.find_objects(ndimage.label(is_white)[0])
#print(regions)
#print(len(regions))

# Ueber identifizierte Regionen iterieren
for region in regions:
    #print(type(region))
    # Aktuelle Region grau (Farbwert=0.5) einfaerben; nur zur Kontrolle und Visualisierung
    """
    img1 = img.copy()
    img1[region] = 0.5
    plt.imshow(img1, cmap='gray')
    plt.show()
    """

    # Rand des Rechengebiets hinzufuegen
    domain = tuple([slice(r.start - 1, r.stop + 1) for r in region])

    # Problem durch Loesung der Laplace-Gleichung loesen; TODO: eigenen Code zur Loesung der Problems einfuegen
    #sol = solve_problem(img, domain)
    
    print(img[domain].shape)

    # Loesung der Region zuordnen
    #img[domain] = sol

# Originalbild und rekonstruiertes Bild darstellen und anzeigen
# fig = plt.figure()
# ax = [fig.add_subplot(121 + i) for i in range(2)]
# ax[0].imshow(img_orig, cmap='gray')
# ax[0].set_title("Orignal")
# ax[1].imshow(img, cmap='gray')
# ax[0].set_title("Reconstruction")
# plt.show()
