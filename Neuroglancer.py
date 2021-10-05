import neuroglancer
import numpy as np
import zarr


if __name__ == "__main__":
    #neuroglancer.set_server_bind_address('10.3.9.26')
    viewer = neuroglancer.Viewer()

    f1 = zarr.open('/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/01.zarr')
    f2 = zarr.open('/Users/sayantaneebiswas/PycharmProjects/proliferation-and-track/Data/02.zarr')
    resolution = [1, 1.75, 1, 1]
    dims = neuroglancer.CoordinateSpace(
                names=['t', 'z', 'y', 'x'],
                units='nm',
                scales=resolution)

    with viewer.txn() as s:
        for ds_name in ['raw', 'GT']:
            data = f2[ds_name]
            vol = neuroglancer.LocalVolume(
                data=data,
                voxel_offset=[0, ] * 4,
                dimensions=dims)
            s.layers.append(name=ds_name, layer=vol, visible=True)

    print(viewer)
    input()







