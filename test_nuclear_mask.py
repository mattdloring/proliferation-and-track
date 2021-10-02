import imageio
from create_nuclear_mask import create_nuclear_mask


def test_case():
    txyz_coords = [[5, 5, 5, 5]]
    resolution = [1, 1, 1, 2]
    shape = (10, 10, 10, 10)
    gaussian_size = 1
    labels = create_nuclear_mask(
        txyz_coords,
        resolution,
        shape,
        gaussian_size)

    imageio.imsave('test_mask_5.jpg', labels[5][5] * 255)
    imageio.imsave('test_mask_6.jpg', labels[6][5] * 255)

    assert labels.shape == shape, labels.shape
    assert labels[5, 5, 5, 5] == 1.0, labels[5, 5, 5, 5]
    assert labels[6, 6, 6, 6] == 0.0, labels[6, 6, 6, 6]
    assert labels[5, 6, 6, 6] > 0.0, labels[5, 6, 6, 6]
    assert labels[5, 6, 6, 6] < 1.0, labels[5, 6, 6, 6]


if __name__ == "__main__":
    test_case()
