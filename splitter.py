import numpy as np
from PIL import Image
import scipy.spatial.distance as SSD
import math
import networkx as nx
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
from math import log10, floor
from decimal import *
from math import sqrt

getcontext().prec = 28
def shuflle2dArray(np_array):
    np_array = np.array(np_array)
    arr_shape = np_array.shape
    new_arr = np_array.reshape(arr_shape[0]*arr_shape[1],arr_shape[-3],arr_shape[-2],arr_shape[-1])
    np.random.shuffle(new_arr)
    arr = new_arr.reshape(arr_shape)
    np_array = arr
    return np_array

def pieces(image, width_num, height_num):
    images = [[None for _ in range(width_num)] for _ in range(height_num)]
    width = len(image[0,:])
    height = len(image[:,0])
    split_height = height // height_num
    split_width = width // width_num
    y=0
    i = 0
    for i in range(0, height_num):
        x = 0
        for j in range(0, width_num):
            temp_img = image[y:y+split_height,x:x+split_width,:]
            images[i][j] = temp_img
            x += split_width
        y += split_height
    images = shuflle2dArray(images)
    return images


def edges(np_image, target) :
    target_size = target
    shape_ar = np.array(np_image).shape
    result_edge = [[['#' for _ in range(4)] for _ in range(shape_ar[1])] for _ in range(shape_ar[0])]
    for x in range(0, len(np_image)) :
        for y in range(0,len(np_image[x])):
            north = np_image[x][y][0:target_size,:,:]
            south = np_image[x][y][-target_size:,:,:]
            west = np_image[x][y][:, 0:target_size, :]
            east = np_image[x][y][:, -target_size:, :]
            result_edge[x][y][0] = north
            result_edge[x][y][1] = east
            result_edge[x][y][2] = south
            result_edge[x][y][3] = west
    return result_edge


def compatibility(np_pieces):
    shape_results = (len(np_pieces),len(np_pieces[0]),4,len(np_pieces),len(np_pieces[0]))
    results = np.empty(shape_results)
    results.fill(np.nan)
    for x in range(0, len(np_pieces)):
        for y in range(0,len(np_pieces[x])):
            print('Piece ' + ' ' + str(x) + ' ' + str(y) + ' finished processing...')
            for z in range(4):
                for xx in range(0, len(np_pieces)):
                    for yy in range(0, len(np_pieces[x])):
                        if x != xx or y != yy:
                            zz = getInverse(z)
                            results[x][y][z][xx][yy] = get_score(np_pieces[x][y][z],np_pieces[xx][yy][zz],z)
    return results

#Return the relevant edge position given a position
def getInverse(p):
    if p == 0:
        return 2
    if p == 1:
        return 3
    if p == 2:
        return 0
    if p == 3:
        return 1

def get_best_match(piece_X, piece_Y, results, edge):
    piece_result = results[piece_X][piece_Y][edge]
    idx = np.argsort(piece_result.ravel())[:2]
    x, y = np.unravel_index(idx[0], piece_result.shape)
    xx, yy = np.unravel_index(idx[1], piece_result.shape)
    score = piece_result[x][y] / piece_result[xx][yy]
    return (x,y,score)

def get_score(piece1, piece2, edge):
    piece1 = matplotlib.colors.rgb_to_hsv(piece1 / float(256))
    piece2 = matplotlib.colors.rgb_to_hsv(piece2 / float(256))
    dummy_gradients = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0],[0, 0, -1]]
    if edge == 0:
        grad_p1 = abs(piece1[0, :, :] - piece1[1, :, :])
        grad_p2 = abs(piece2[-1, :, :] - piece2[-2, :, :])
        grad_p1p2 = abs(piece2[-1, :, :] - piece1[0, :, :])

    elif edge == 1:
        grad_p1 = abs(piece1[:,-1,:] - piece1[:,-2,:])
        grad_p2 = abs(piece2[:, 0, :] - piece2[:, 1, :])
        grad_p1p2 = abs(piece2[:, 0, :] - piece1[:, -1, :])


    elif edge == 2:
        grad_p1 = abs(piece1[-1, :, :] - piece1[-2, :, :])
        grad_p2 = abs(piece2[0, :, :] - piece2[1, :, :])
        grad_p1p2 = abs(piece2[0, :, :] - piece1[-1, :, :])


    elif edge == 3:
        grad_p1 = abs(piece1[:,0, :] - piece1[:, 1, :])
        grad_p2 = abs(piece2[:, -1, :] - piece2[:, -2, :])
        grad_p1p2 = abs(piece2[:, -1, :] - piece1[:, 0, :])

    else:
        raise ValueError('Edge number out of range')

    gr_p1_mean = np.mean(grad_p1)
    gr_p2_mean = np.mean(grad_p2)

    gr_diff_p1_mean = abs(grad_p1p2 - gr_p1_mean)
    gr_diff_p2_mean = abs(grad_p1p2 - gr_p2_mean)

    grad_p1_dummy = np.append(grad_p1, dummy_gradients, axis=0)
    grad_p2_dummy = np.append(grad_p2, dummy_gradients, axis=0)

    p1_cov = np.cov(grad_p1_dummy, rowvar=False)
    p2_cov = np.cov(grad_p2_dummy, rowvar=False)

    p1_cov_inv = np.linalg.inv(p1_cov)
    p2_cov_inv = np.linalg.inv(p2_cov)

    mahalanobis_distp1p2 = sqrt(np.sum(np.dot(np.dot(gr_diff_p1_mean, p1_cov_inv),np.transpose(gr_diff_p1_mean))))
    mahalanobis_distp2p1 = sqrt(np.sum(np.dot(np.dot(gr_diff_p2_mean, p2_cov_inv),np.transpose(gr_diff_p2_mean))))

    return(mahalanobis_distp1p2 + mahalanobis_distp2p1)

"""
Determine if two piece are best matches to one another (Best buddy)
@return True if best buddy else False
"""
def is_best_match(x,y,xx,yy, results, edge):
    x_match,y_match = get_best_match(x,y,results,edge)[:2]
    return (x_match,y_match) == (xx,yy)

def print_graph(graphs):
    for graph in graphs:
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, node_size=600)
        nx.draw_networkx_edges(graph, pos, edge_color='black', width=3)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.axis('off')
        print(graph.adj.items())
        plt.show()


def reconstruct(graphs, results):
    res_shape = results.shape
    h = res_shape[0]
    w = res_shape[1]
    solution = [[['#' for _ in range(4)] for _ in range(res_shape[1])] for _ in range(res_shape[0])]
    for graph in graphs:
        old_w = 1000000
        for (u, v, wt) in graph.edges.data('weight'):
            edge = graph.get_edge_data(u, v)['edge']
            print('edge ',edge)
            if is_best_match(u[0],u[1],v[0],v[1],results,edge):
                solution[u[0]][u[1]][edge] = v
                solution[v[0]][v[1]][edge] = u
    return solution

def display_results(results,images):
    img_final = np.zeroes((1000,668,3))
    for x in range(results):
        for y in range(results):
            for z in range(4):
                if results[x][y][z] != 0:
                    target_x = results[x][y][z][0]
                    target_Y = results[x][y][z][1]
                    if z == 0:
                        insert_image(img_final, images[x][y], edge)

                    if z == 1:
                        insert_image
                    if z == 2:
                        insert_image
                    if z ==3:
                        insert_image
                    results[target_x][target_Y][z] = 0

def graph_create(results):
    graphs = []
    graph = nx.Graph()
    for x in range(0, len(results)):
        for y in range(0, len(results[x])):
            graph = nx.Graph()
            for z in range(4):
                match_x, match_y, match_score = get_best_match(x,y,results,z)
                # Check if edge already exists, if it does only replace current edge if match_score is smaller than existing score (this can happen at edges)
                if graph.get_edge_data((x,y),(match_x,match_y)) is not None and graph.get_edge_data((x,y),(match_x,match_y))['weight'] < match_score:
                    print('Edge already exists and has a lower score for piece', x, y)
                else:
                    graph.add_edge((x,y), (match_x,match_y), weight=round(match_score,5), edge=z)
            graphs.append(graph)
    return graphs


