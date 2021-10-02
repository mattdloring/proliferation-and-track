import numpy as np
from scipy.ndimage import gaussian_filter


def create_nuclear_mask(txyz_coords, resolution, shape, gaussian_size):
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
    arr = np.zeros(shape)
    # step 1: create a binary mask with a 1 at every coordinate
    for coord in txyz_coords:
        arr[coord[0], coord[1], coord[2], coord[3]] = 1.0
    # step 2: Feed that mask into scipy.ndimage.gaussian_filter()
    # for t in range(arr.shape[0]):
    #    arr[t] = gaussian_filter(arr[t], gaussian_size)
    labels = gaussian_filter(arr, sigma=(0, gaussian_size, gaussian_size, gaussian_size))
    # step 3: Normalize the labels to [0, 1] (make sure the max is 1)
    max_value = np.max(labels)
    print(max_value)
    labels = labels / max_value
    return labels


