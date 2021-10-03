from utils import convert_to_n5

import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np

import zarr

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Dataset_3DT(Dataset):
    def __init__(self, input_z_arr_path):
        ## expects zarr file with at least raw data

        self.input_z_arr_path = input_z_arr_path
        self.datasource = zarr.open(self.input_z_arr_path)
        self.dims = self.datasource['raw'].shape

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

        self._ground_truth_mask = gp.ArrayKey('GT_mask')
        self.gt_source_mask = gp.ZarrSource(
            self.input_z_arr_path,
            {self._ground_truth_mask: 'GT_mask', }
        )

        self.comb_source = ((self.raw_source, self.gt_source + self.gt_source_mask) + gp.MergeProvider())

        random_location = gp.RandomLocation(mask=self._ground_truth_mask, min_masked=0.8)
        ## recject node

        self.basic_request = gp.BatchRequest()
        self.basic_request[self._raw] = gp.Roi((0, 0, 0, 0), (1, 32, 256, 256))
        self.basic_request[self._ground_truth] = gp.Roi((0, 0, 0, 0), (1, 32, 256, 256))
        self.basic_request[self._raw]
        self.basic_request[self._ground_truth]

        self.random_sample = self.comb_source + random_location

    def __getitem__(self):
        with gp.build(self.random_sample):
            batch = self.random_sample.request_batch(self.basic_request)
        return batch[self._raw].data, batch[self._ground_truth].data



if __name__ == '__main__':
    zarr_path = '/home/loringm/Downloads/SIMULATED_DATASET/01/data.n5'

    outputs = Dataset_3DT(zarr_path).__getitem__()
    print(outputs)