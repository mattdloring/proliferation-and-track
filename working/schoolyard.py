'''

We go to school here.

'''

from skimage.segmentation import relabel_sequential

import networkx as nx
import numpy as np

import itertools


class Tracker:
    def __init__(self, image1, image2, plot=False):
        '''
        expects two images
        calculates best match of cells and returns various celly stats
        '''
        self.image1 = image1
        self.image2 = image2
        self.plot = plot

        self.weight_mtx = self.calculate_diff_matrix(self.image1, self.image2)
        self.layer1_depth = self.weight_mtx.shape[0]

    def find_match(self):
        self.network, self.matches = self.create_network(self.weight_mtx)

    def create_network(self):
        def multilayered_graph(*subset_sizes):
            extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
            layers = [range(start, end) for start, end in extents]
            G = nx.Graph()
            for (i, layer) in enumerate(layers):
                G.add_nodes_from(layer, layer=i)
            for layer1, layer2 in nx.utils.pairwise(layers):
                G.add_edges_from(itertools.product(layer1, layer2))
            return G
        raw_weights = self.weight_mtx.ravel()

        lays = [range(start, end) for start, end in nx.utils.pairwise([self.weight_mtx.shape])]
        G = nx.Graph()
        for (i, layer) in enumerate(lays):
            G.add_nodes_from(lays, layer=i)
        for layer1, layer2 in nx.utils.pairwise(lays):
            G.add_edges_from(itertools.product(layer1, layer2))
        G = multilayered_graph(self.weight_mtx.shape[0], self.weight_mtx.shape[1])
        edges = list(G.edges)
        weights = {edges[j]: raw_weights[j] for j in range(len(edges))}
        nx.set_edge_attributes(G, values=weights, name='weight')
        return G, nx.algorithms.bipartite.matching.minimum_weight_full_matching(G)

    @staticmethod
    def calculate_diff_matrix(img1, img2, init_val=50):
        for image in [img1, img2]:
            try:
                assert (image.min() == 0)
            except AssertionError:
                print('image must contain background class')

        img1 = np.int32(img1)
        img2 = np.int32(img2)

        img1 = relabel_sequential(img1)[0]
        img2 = relabel_sequential(img2)[0]

        unique_img1 = np.unique(img1)
        unique_img2 = np.unique(img2)

        diff_matrix = np.ones([len(unique_img1), len(unique_img2)]) * init_val

        for y in unique_img2:
            for x in unique_img1:
                # diff = ((np.array(np.where(img1 == x)) - np.array(np.where(img2 == y))) ** 2).sum() # previous working solution, doesnt handle arbirtrary array sizes
                diff = ((np.mean(np.where(img1 == x), axis=1) - np.mean(np.where(img2 == y), axis=1)) ** 2).sum()

                diff_matrix[x, y] = diff

        # this is a bit janky but it handles shifting the background
        relatively_big_number = diff_matrix.max() ** 2
        diff_matrix[0, :] = relatively_big_number
        diff_matrix[:, 0] = relatively_big_number

        return diff_matrix


def clean_match_dictionary(matches, first_layer_depth):
    # more fuckery to keep the background from swapping
    for key in matches.copy().keys():
        if matches[key] <= first_layer_depth:
            del matches[key]
    matches[0] = first_layer_depth
    return matches
