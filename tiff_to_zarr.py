from utils import convert_to_n5

if __name__ == '__main__':
    zarrfile1 = '/Users/sayantaneebiswas/PycharmProjects/Data/01.zarr'
    zarrfile2 = '/Users/sayantaneebiswas/PycharmProjects/Data/02.zarr'
    image1 = '/Users/sayantaneebiswas/Downloads/101620-19hsp-H2A-01_embryo_01.tif'
    image2 = '/Users/sayantaneebiswas/Downloads/101620-19hsp-H2A-01_embryo_02.tif'
    resolution = [1,1.75,1,1]
    key = 'raw'
    convert_to_n5(image1, zarrfile1, key, resolution)
    convert_to_n5(image2, zarrfile2, key, resolution)
