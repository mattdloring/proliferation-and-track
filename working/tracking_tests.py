from tracking import find_match, map_matches
import numpy as np
from scipy import ndimage
import tifffile
import zarr


zarr_dir = '/home/loringm/Downloads/SIMULATED_DATASET/01/data.n5'
data = zarr.open(zarr_dir)

imgs = data['GT'][:, 30, 144:400, 144:400]

bnr_images = [np.array(img > 0, dtype=np.int8) for img in imgs]
labeled_images = [ndimage.label(bnrimage)[0] for bnrimage in bnr_images]

remapped_images = []
for i in range(len(bnr_images)):
    if i is not 0:
        match_dic, l1_shape = find_match(bnr_images[i-1], bnr_images[i])

        predicted_image = map_matches(bnr_images[i], match_dic, l1_shape)

        remapped_images.append(predicted_image)

fin = np.array(remapped_images)
tifffile.imsave('mywonderful_remap.tiff', fin)