import scipy.ndimage
import numpy as np

def create_nuclear_mask(txyz_coords,resolution,shape,gaussian_size):
    ''' Create a gaussian mask around each txyz coordinate in the dataset
    Inputs: 
        txyz_coords: list(list(int))
            A list of coordinates (t, x, y, z)

        resolution: list(int)
            Approximate pixel resolution in (t, x, y, z) (where t is 1)

        shape: list(int)
            Total shape of the dataset (t, x, y, z) - number of pixels in each dimension

        gaussian_size: float
            Parameter to determine the size of the gaussian mask

    Returns:
        An array of the same shape as the data, with data type float, with gaussian blobs at each coordinate.
    '''
    # step 0: create a np array of zeros (of the shape of the mask)

    # step 1: create a binary mask with a 1 at every coordinate

    # step 2: Feed that mask into scipy.ndimage.gaussian_filter()

    mask_arr = np.zeros(shape)
    for co in txyz_coords:
        t = co[0]
        x = co[1]
        y = co[2]
        z = co[3]
        mask_arr[t][x][y][z] = 1

    sigma = [0,gaussian_size/resolution[1],gaussian_size/resolution[2],gaussian_size/resolution[3]]
    mask_gaussian = scipy.ndimage.gaussian_filter(mask_arr, sigma=sigma)

    max_val = np.max(mask_gaussian)
    mask_gaussian = mask_gaussian/max_val
    return mask_gaussian










