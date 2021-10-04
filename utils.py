import zarr
import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import os
import argparse
import logging
try:
    import gunpowder as gp

    import torch.nn as nn
    import torch
    from torch.utils.data import Dataset, DataLoader
except ModuleNotFoundError:
    print('some stuff not available')


logger = logging.getLogger(__name__)


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
    print(f.keys())
    f[key] = raw
    # x, y, z, t for N5
    if resolution is None:
        resolution = np.ones(4)

    f[key].attrs['resolution'] = resolution
    f[key].attrs['offset'] = [0, 0, 0, 0]

try:
    class Dataset_3DT(Dataset):
        def __init__(self, input_z_arr_path):
            ## expects zarr file with at least raw data

            self.input_z_arr_path = input_z_arr_path
            self.datasource = zarr.open(self.input_z_arr_path)

            self.build_pipelines()

        def build_pipelines(self):
            self._raw = gp.ArrayKey('raw')
            self.raw_source = gp.ZarrSource(
                self.input_z_arr_path,
                {self._raw: 'raw', }
            )

            self._ground_truth = gp.ArrayKey('GT')
            self.gt_source = gp.ZarrSource(
                self.input_z_arr_path,
                {self._ground_truth: 'GT', }
            )

            self.comb_source = ((self.raw_source, self.gt_source) + gp.MergeProvider())

            random_location = gp.RandomLocation()

            self.basic_request = gp.BatchRequest()
            self.basic_request[self._raw] = gp.Roi((0, 0, 0, 0),
                                                   (1, 1, self.datasource['raw'].shape[2], self.datasource['raw'].shape[3]))
            self.basic_request[self._ground_truth] = gp.Roi((0, 0, 0, 0), (
            1, 1, self.datasource['raw'].shape[2], self.datasource['raw'].shape[3]))

            self.random_sample = self.comb_source + random_location

        def __getitem__(self):
            with gp.build(self.random_sample):
                batch = self.random_sample.request_batch(self.basic_request)

            return batch[self._raw].data, batch[self._ground_truth].data
except NameError:
    print('also other stuff not available')