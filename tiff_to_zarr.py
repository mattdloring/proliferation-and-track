import zarr
import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import os


def convert_to_n5(input_path, outfile, key='raw', resolution=None):
    '''
    :param input_path: input path can be a directory, where we will search for all tifs in directory,
        or it can be a single tif file
    :param outfile:
    :param key:
    :param resolution:
    :return:
    '''
    assert os.path.exists(input_path)
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, '*.tif')))
    elif input_path.endswith('tiff') or input_path.endswith('tif'):
        files = [input_path]
    else:
        raise NotImplemented('Only takes a directory or tif')

    assert len(files) > 0, f"No tif files in {input_path}"
    print(files)

    raw = np.array([imread(f) for f in tqdm(files)])
    raw = np.squeeze(raw)
    print(raw.shape)
    print(raw.dtype)
    print(raw.min(), raw.max())

    print(outfile)
    f = zarr.open(outfile, 'a')
    print("GROUND TRUTH " + str(f['GT'].shape))
    f[key] = raw
    # x, y, z, t for N5
    if resolution is None:
        resolution = np.ones(4)

    f[key].attrs['resolution'] = resolution
    f[key].attrs['offset'] = [0, 0, 0, 0]


if __name__ == '__main__':
    zarrfile1 = '/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/01.zarr'
    zarrfile2 = '/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/02.zarr'
    image1 = '/Users/sayantaneebiswas/Downloads/101620-19hsp-H2A-01_embryo_01.tif'
    image2 = '/Users/sayantaneebiswas/Downloads/101620-19hsp-H2A-01_embryo_02.tif'
    resolution = [1,1.75,1,1]
    key = 'raw'
    convert_to_n5(image1, zarrfile1, key, resolution)
    convert_to_n5(image2, zarrfile2, key, resolution)
