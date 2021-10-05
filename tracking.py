import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from skimage.segmentation import relabel_sequential

import itertools


def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G


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

    ## actually this doesnt fix it
    ## this works for N x N+1 but not N+1 x N
    relatively_big_number = diff_matrix.max() ** 2
    diff_matrix[0, :] = relatively_big_number
    diff_matrix[:, 0] = relatively_big_number

    return diff_matrix


def find_match(input_image_1, input_image_2, plot=False):
    #### FIND MATCH NEEDS POST PROCESSING
    #### RIGHT NOW THIS RETURNS BEST MATCHES
    #### POST PROCESSING WILL ALSO NEED THE WEIGHTS OF THE MATCHES
    #### ABHORRENT WEIGHTS SHALL BE PUNISHED
    weight_mtx = calculate_diff_matrix(input_image_1, input_image_2)
    l1_shape = weight_mtx.shape[0]
    raw_weights = weight_mtx.ravel()

    lays = [range(start, end) for start, end in nx.utils.pairwise([weight_mtx.shape])]

    G = nx.Graph()
    for (i, layer) in enumerate(lays):
        G.add_nodes_from(lays, layer=i)
    for layer1, layer2 in nx.utils.pairwise(lays):
        G.add_edges_from(itertools.product(layer1, layer2))

    G = multilayered_graph(weight_mtx.shape[0], weight_mtx.shape[1])

    edges = list(G.edges)
    weights = {edges[j]: raw_weights[j] for j in range(len(edges))}

    pos = nx.multipartite_layout(G, subset_key="layer")

    nx.set_edge_attributes(G, values=weights, name='weight')

    labels = nx.get_edge_attributes(G, 'weight')
    if plot:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, horizontalalignment='left', verticalalignment='top')
        nx.draw(G, pos, with_labels=True)
    return nx.algorithms.bipartite.matching.minimum_weight_full_matching(G), l1_shape


def map_matches(mapped_array, match_dictionary, first_layer_depth):
    ### adapted from https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key ###
    u,inv = np.unique(mapped_array,return_inverse = True)
    return np.array([match_dictionary[x]-first_layer_depth for x in u])[inv].reshape(mapped_array.shape)

