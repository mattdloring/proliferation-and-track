import zarr
import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread
import os
import argparse
import logging
import gunpowder as gp

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader


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

