#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#%%
img = plt.imread("Bild.jpg")

#%%
print(type(img))

print(img.shape)
# %%

gauss = plt.imread("gauss.png")
# %%
print(gauss.shape)
# %%

fig = plt.figure()
ax = [fig.add_subplot(121)]
ax.append(fig.add_subplot(122))

ax[0].imshow(img[:,:,0], cmap='gray')
ax[1].imshow(img)

# %%

print(img[:5, :5,:])
print(gauss[:5, :5,:])

# %%
bwgauss = gauss[..., 0]
# %%
plt.imshow(bwgauss, cmap='gray',)
# %%
# Suche die Weißen Rechtecke

is_white = bwgauss == 1.
regions = ndimage.find_objects(ndimage.label(is_white)[0])
print(regions)

# %%
for region in regions:
#     imgl = bwgauss.copy()
#     imgl[region] = 0.5
#     plt.imshow(imgl, cmap='gray')
#     plt.show()
    # rand des Rechengebiets hinzufügen
    domain = (slice(region[0].start - 1, region[0].stop + 1), 
              slice(region[1].start - 1, region[1].stop + 1))
    plt.imshow(bwgauss[domain], cmap='gray')
    plt.show()
# %%
