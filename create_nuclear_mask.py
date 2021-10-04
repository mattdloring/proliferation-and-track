import numpy as np
import scipy.ndimage
import csv
import pandas as pd
import tifffile
import zarr
from utils import convert_to_n5
from skimage import io

def create_nuclear_mask(tzyx_coords,resolution,shape,gaussian_size):
    ''' Create a gaussian mask around each txyz coordinate in the dataset
    Inputs: 
        txyz_coords: list(list(int))
            A list of coordinates (t, z, y, x)

        resolution: list(int)
            Approximate pixel resolution in (t, z, y, x) (where t is 1)

        shape: list(int)
            Total shape of the dataset (t, z, y, x) - number of pixels in each dimension

        gaussian_size: float
            Parameter to determine the size of the gaussian mask

    Returns:
        An array of the same shape as the data, with data type float, with gaussian blobs at each coordinate.
    '''
    # step 0: create a np array of zeros (of the shape of the mask)

    # step 1: create a binary mask with a 1 at every coordinate

    # step 2: Feed that mask into scipy.ndimage.gaussian_filter()

    mask_arr = np.zeros(shape, dtype=np.float32)
    for co in tzyx_coords:
        t = co[0]
        z = co[3]
        y = co[2]
        x = co[1]
        mask_arr[t][z][y][x] = 1


    sigma = [0,gaussian_size/resolution[1],gaussian_size/resolution[2],gaussian_size/resolution[3]]
    mask_gaussian = scipy.ndimage.gaussian_filter(mask_arr, sigma=sigma)

    max_val = np.max(mask_gaussian)
    mask_gaussian = mask_gaussian/max_val
    return mask_gaussian.astype(np.float32)

def load_csv (filename):
    df = pd.read_csv(filename)
    tzyx_coords = df.iloc[1::, 2:6]
    tzyx_coords = tzyx_coords.to_numpy()-1
    return tzyx_coords


def save_mask(mask, filename):
    print("saving tiff at file" + filename)
    # reorder the dimensions: tzyx -> xyczt
    # transposed_mask = np.transpose(mask, [3,2,1,0])
    print(mask.shape)
    # with_channel = np.expand_dims(mask, 2)
    # print(with_channel.shape)
    opened_zarr = zarr.open(filename, 'w')
    opened_zarr['GT'] = mask
    opened_zarr['GT'].attrs['resolution'] = [1,1.75,1,1]
    opened_zarr['GT'].attrs['offset'] = [0, 0, 0, 0]


if __name__ == "__main__":
    # Configuration
    csv_file1 = '/Users/sayantaneebiswas/Desktop/GT_01.csv'
    csv_file2 = '/Users/sayantaneebiswas/Desktop/GT_02.csv'
    dataset_shape1 = [129, 50, 332, 1024]
    dataset_shape2 = [128, 50, 332, 1007]
    gaussian_size = 5
    zarrfile1 = '/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/01.zarr'
    zarrfile2 = '/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/02.zarr'

    # running code
    csv = load_csv(csv_file1)
    # print(csv)
    mask = create_nuclear_mask(csv, [1, 1.75, 1, 1], dataset_shape1, gaussian_size)
    print('done masking')
    save_mask(mask, zarrfile2)

    csv = load_csv(csv_file2)
    mask = create_nuclear_mask(csv, [1, 1.75, 1, 1], dataset_shape2, gaussian_size)
    print('done masking')
    save_mask(mask, zarrfile2)







