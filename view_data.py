import neuroglancer
import numpy as np
import zarr
import pandas as pd
import itertools

ngid = itertools.count(start=1)

df = pd.read_csv('GT_02_edit.csv')

nodes = []

for index, row in df.iterrows():
    site = [row['T'], row['Z'], row['Y'], row['X']]
    print(site)
    nodes.append(
            neuroglancer.EllipsoidAnnotation(
                center=site,
                radii=(5,5,5,5),
                id=next(ngid)))

viewer = neuroglancer.Viewer()

f = zarr.open('/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/02.zarr')

raw = np.squeeze(f['raw'][:])
res = f['raw'].attrs['resolution']
off = f['raw'].attrs['offset']

mask = np.squeeze(f['GT'][:])
#mask = np.transpose(mask, axes=[0,3,2,1])

print(raw.shape, mask.shape)

dimensions = neuroglancer.CoordinateSpace(
            names=['t', 'x', 'y', 'z'],
            units='nm',
            scales=res,
        )

raw_vol = neuroglancer.LocalVolume(
	data=raw,
	voxel_offset=off,
	dimensions=dimensions)

mask_vol = neuroglancer.LocalVolume(
	data=mask,
	voxel_offset=off,
	dimensions=dimensions)
	
with viewer.txn() as s:
    s.layers['raw'] = neuroglancer.ImageLayer(source=raw_vol)
    s.layers['mask'] = neuroglancer.ImageLayer(source=mask_vol)
    s.layers['nodes'] = neuroglancer.LocalAnnotationLayer(
        dimensions=dimensions,
        annotations=nodes)

print(viewer)
