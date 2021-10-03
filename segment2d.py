from utils import convert_to_n5

import gunpowder as gp
import matplotlib.pyplot as plt
import numpy as np

import zarr

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

assert torch.cuda.is_available()
device = torch.device("cuda")
import logging

logging.basicConfig(level=logging.INFO)

class Dataset2d(Dataset):
    def __init__(self, input_z_arr_path):

        self.input_z_arr_path = input_z_arr_path
        self.datasource = zarr.open(self.input_z_arr_path)
        self.dims = self.datasource['raw'].shape

        self.build_pipelines()

    def build_pipelines(self):
        self._raw_key = gp.ArrayKey('raw')
        self.raw_source = gp.ZarrSource(
            self.input_z_arr_path,
            {self._raw_key: 'raw', },
            {self._raw_key: gp.ArraySpec(interpolatable=True)}

        )

        self._ground_truth_key = gp.ArrayKey('GT')
        self.gt_source = gp.ZarrSource(
            self.input_z_arr_path,
            {self._ground_truth_key: 'GT', },
            {self._ground_truth_key: gp.ArraySpec(interpolatable=True) }
        )

        self._ground_truth_mask = gp.ArrayKey('GT_mask')
        self.gt_source_mask = gp.ZarrSource(
            self.input_z_arr_path,
            {self._ground_truth_mask: 'GT_mask', },
            {self._ground_truth_mask: gp.ArraySpec(interpolatable=True)}
        )

        self.raw_test_key = gp.ArrayKey('raw_test')
        self.raw_test_source = gp.ZarrSource(
            self.input_z_arr_path,
            {self.raw_test_key: 'raw_test', },
            {self.raw_test_key: gp.ArraySpec(interpolatable=True)}
        )
        self.raw_test_GT_key = gp.ArrayKey('raw_test_GT')
        self.raw_test_GT_source = gp.ZarrSource(
            self.input_z_arr_path,
            {self.raw_test_GT_key: 'raw_test_GT', },
            {self.raw_test_GT_key: gp.ArraySpec(interpolatable=True)}
        )

        self.comb_source = ((self.raw_source, self.gt_source_mask) + gp.MergeProvider())
        self.test_source = ((self.raw_test_source, self.raw_test_GT_source) + gp.MergeProvider())
        # random_location = gp.RandomLocation(mask=self._ground_truth_mask, min_masked=0.8)
        random_location = gp.RandomLocation()

        self.basic_request = gp.BatchRequest()
        self.basic_request[self._raw_key] = gp.Roi((0, 0, 0, 0), (1, 1, 256, 256))
        self.basic_request[self._ground_truth_mask] = gp.Roi((0, 0, 0, 0), (1, 1, 256, 256))

        self.test_request = gp.BatchRequest()
        self.test_request[self.raw_test_key] = gp.Roi((0, 0, 0, 0), (1, 1, 256, 256))
        self.test_request[self.raw_test_GT_key] = gp.Roi((0, 0, 0, 0), (1, 1, 256, 256))

        simple_augment = gp.SimpleAugment(transpose_only=[2, 3])

        normalize = gp.Normalize(self.comb_source, factor=1/1023 )

        intensity_augment = gp.IntensityAugment(
            self._raw_key,
            scale_min=0.9,
            scale_max=1.1,
            shift_min=-0.1,
            shift_max=0.1)
        noise_augment = gp.NoiseAugment(self._raw_key)

        self.random_sample = self.comb_source + random_location + simple_augment

        self.random_test_sample = self.test_source + random_location
        # self.random_sample = self.comb_source + random_location + simple_augment

        # self.random_sample = self.comb_source + random_location

    def __getitem__(self):
        max_val = 0
        while max_val == 0:
            with gp.build(self.random_sample):
                batch = self.random_sample.request_batch(self.basic_request)

            max_val = np.max(batch[self._ground_truth_mask].data)

        return batch[self._raw_key].data, batch[self._ground_truth_mask].data

    def get_test_item(self):
        max_val = 0
        while max_val == 0:
            with gp.build(self.random_test_sample):
                batch = self.random_test_sample.request_batch(self.test_request)

            max_val = np.max(batch[self.raw_test_GT_key].data)

        return batch[self.raw_test_key].data, batch[self.raw_test_GT_key].data

class UNet(nn.Module):
    """ UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                             nn.ReLU())

        # upsampling via transposed 2d convolutions

    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1,
                 final_activation=None):
        super().__init__()

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = 4

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation, nn.Module), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList([self._conv_block(in_channels, 16),
                                      self._conv_block(16, 32),
                                      self._conv_block(32, 64),
                                      self._conv_block(64, 128)])
        # the base convolution block
        self.base = self._conv_block(128, 256)
        # modules of the decoder path
        self.decoder = nn.ModuleList([self._conv_block(256, 128),
                                      self._conv_block(128, 64),
                                      self._conv_block(64, 32),
                                      self._conv_block(32, 16)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(256, 128),
                                         self._upsampler(128, 64),
                                         self._upsampler(64, 32),
                                         self._upsampler(32, 16)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(16, out_channels, 1)
        self.activation = final_activation

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def train(model, data_input, optimizer, loss_function,
          epoch, log_interval=100, log_image_interval=20, tb_logger=None):
    # set the model to train mode
    model.train()

    # move input and target to the active device (either cpu or gpu)
    x, y = data_input
    x = torch.Tensor(x.astype(np.int16))
    y = torch.Tensor(y.astype(np.int16))

    x, y = x.to(device), y.to(device)
    # print(x.shape)

    # zero the gradients for this iteration
    optimizer.zero_grad()

    # apply model and calculate loss
    prediction = model(x)
    loss = loss_function(prediction, y)

    # backpropagate the loss and adjust the parameters
    loss.backward()
    optimizer.step()

    # log to tensorboard
    if tb_logger is not None:
        step = epoch
        tb_logger.add_scalar(tag='train_loss', scalar_value=loss.item(), global_step=step)


def validate(model, data_input, loss_function, metric, epoc_l=None, step=None, tb_logger=None):
    # set model to eval mode
    model.eval()
    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        x, y = data_input

        x = torch.Tensor(x.astype(np.int16))
        y = torch.Tensor(y.astype(np.int16))

        # iterate over validation loader and update loss and metric values
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        val_loss += loss_function(prediction, y).item()
        val_metric += metric(prediction, y).item()

    # normalize loss and metric
    # val_loss /= epoc_l
    # val_metric /= epoc_l

    if tb_logger is not None:
        assert step is not None, "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(tag='val_metric', scalar_value=val_metric, global_step=step)

    print('\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n'.format(val_loss, val_metric))


class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return (2 * intersection / denominator.clamp(min=self.eps))


if __name__ == '__main__':
    zarr_path = '/home/loringm/Downloads/SIMULATED_DATASET/01/data.n5'

    print(Dataset2d(zarr_path).__getitem__()[0].shape)
    
