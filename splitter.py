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
    #Image.fromarray(image).show()
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
    results[:] = 1000
    for x in range(0, len(np_pieces)):
        for y in range(0,len(np_pieces[x])):
            print('Piece ' + ' ' + str(x) + ' ' + str(y) + ' finished processing...')
            for z in range(4):
                for xx in range(0, len(np_pieces)):
                    for yy in range(0, len(np_pieces[x])):
                        if x != xx or y != yy:
                            zz = getInverse(z)
                            results[x][y][z][xx][yy] = get_score(np_pieces[x][y][z],np_pieces[xx][yy][zz],z)
    #for x, elem in enumerate(results):
        #results[x].sort()
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
    print(piece_result)
    print(piece_result.argmin())
    return np.unravel_index(piece_result.argmin(), piece_result.shape)

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

def print_results(results):
    piece_num = 0
    results_matrix = np.full((len(results), 16), -1, dtype='float')
    for x, result in enumerate(results):
        edge_num = x % 4
        if x % 4 == 0 and x != 0:
            piece_num += 1
        print('Piece number ' + str(piece_num) + ' edge number ' + str(edge_num))
        print(np.array(result))
    for x in range(0, len(results)):
        for y in range(0, len(results[x])):
           edge = results[x][y][1]
           score = results[x][y][0]
           results_matrix[x,edge] = score
    print(results_matrix)




def reconstruct(results, np_pieces):
    np_pieces = np.array(np_pieces)
    total = np_pieces.shape[0]
    h = np_pieces[0].shape[0]
    w = np_pieces[0].shape[1]
    target_width = math.sqrt(total) * w
    target_height = math.sqrt(total) * h
    x = 0
    y = 0
    result = Image.new('RGB', (int(target_width), int(target_height)))
    img = Image.fromarray(np_pieces[0])
    w, h = img.size
    result.paste(img, (x, y))
    for x in range(0, len(results)):
        score, piece = results[x][0]
        position = piece % 4
        piece = int(piece  / 4)
        print(str(piece) + ' ' + str(piece))
        target_image = np_pieces[piece]
        target_image = Image.fromarray(target_image)
        if position == 3 and piece == 2:
            result.paste(target_image, (w, 0))
        if position == 0 and piece == 1:
            result.paste(target_image, (0, h))
        if position == 0and piece == 3:
            result.paste(target_image, (w, h))
    result.show()

# Remove any result < 10^-6 and equal to nan
# Return the smallest result divided by the second smallest in order to discern good results from inconclusive ones
def result_cleaner(results):
    r_shape = results.shape
    d = defaultdict(lambda: defaultdict(dict))
    for x in range(0, len(results)):
        for y in range(0, len(results[x])):
            for z in range(4):
                s = results[x][y][z].shape
                res = np.argsort(np.ravel(results[x][y][z]))[:2]
                res_form = np.unravel_index(res, s)
                lX = res_form[0][0]
                lY = res_form[0][1]
                lXX = res_form[1][0]
                lYY = res_form[1][1]
                d[x][y][z] = (results[x][y][z][lX][lY]/results[x][y][z][lXX][lYY], lX, lY)
    return d

def print_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=300)
    nx.draw_networkx_edges(graph, pos, edge_color='black', width=3)
    labels = nx.get_edge_attributes(graph, 'weight')
    print(labels)
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.axis('off')
    plt.show()





def graph_create(results):
    graph = nx.Graph()
    for x in range(0, len(results)):
        for y in range(0, len(results[x])):
            for z in range(4):
                graph.add_edge((x,y), (results[x][y][z][1:]), weight=results[x][y][z][0])
    print(nx.shortest_path(graph, source=(0,0), target=(1,1)))
    return graph



def graph_init(results):
    graphs = []
    nodes = [x for x in range(0,len(results))]
    count = 0
    piece_num = 0
    graph = nx.Graph()
    start = 0
    end = 10
    for y in range(0, len(results)):
        graph = nx.Graph()
        start = y*4
        end = start + 4
        for x in range(start, end):
            print(x)
            score, position = results[x]
            piece_target = position // 4
            edge_target = position % 4
            piece_num = y
            edge_num = x
            print('Piece number ' + str(piece_num) + ' ' + ' to piece num ' + str(piece_target))
            graph.add_edge(piece_num, piece_target, weight=score, object={edge_num:edge_target})
        graphs.append(graph)
    for graph in graphs:
        pos = nx.spring_layout(graph)
        #e1 = [(u, v) for (u, v, d) in graph.edges(data=True) if d['object'][0] == 0]
        #e2 = [(u, v) for (u, v, d) in graph.edges(data=True) if d['object'][0] == 1]
        #e3 = [(u, v) for (u, v, d) in graph.edges(data=True) if d['object'][0] ==2]
        #e4 = [(u, v) for (u, v, d) in graph.edges(data=True) if d['object'][0] == 3]
        nx.draw_networkx_nodes(graph, pos, node_size=300)
        nx.draw_networkx_edges(graph, pos,edge_color='b',width=6)
        #nx.draw_networkx_edges(graph, pos,edgelist=e1, edge_color='b',width=6)
        #nx.draw_networkx_edges(graph, pos, edgelist=e2,edge_color='r',width=6)
        #nx.draw_networkx_edges(graph, pos, edgelist=e3,edge_color='g',width=6)
        #nx.draw_networkx_edges(graph, pos, edgelist=e4,edge_color='y',width=6)
        labels = nx.get_edge_attributes(graph, 'weight')
        labels2 = nx.get_edge_attributes(graph, 'object')
        nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.axis('off')
        plt.show()
