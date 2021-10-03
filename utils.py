import zarr
import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import os
import argparse
import logging

logger = logging.getLogger(__name__)


def convert_to_n5(directory, outfile, key='raw', resolution=None):
    assert os.path.exists(directory)
    files = sorted(glob.glob(os.path.join(directory, '*.tif')))
    assert len(files) > 0, f"No tif files in {directory}"
    logger.debug(files)

    raw = np.array([imread(f) for f in tqdm(files)])
    logger.debug(raw.shape)
    logger.debug(raw.dtype)
    logger.debug(raw.min(), raw.max())

    f = zarr.open(outfile, 'a')
    f[key] = raw
    # x, y, z, t for N5
    if resolution is None:
        resolution = np.ones(4)

    f[key].attrs['resolution'] = resolution
    f[key].attrs['offset'] = [0, 0, 0, 0]