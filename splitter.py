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

def convert_coord(y,x,piece_num):
    return y*piece_num+x

def decode_coord(coord, piece_num):
    return np.unravel_index(int(coord), (piece_num*piece_num,piece_num*piece_num))

def insert_piece(y,x,z,arr, y_tar, x_tar):
    piece_num = len(arr)
    coord = convert_coord(y_tar,x_tar,piece_num)
    if z == 0:
        if at_edge(y,x,z, piece_num):
            arr = move_down(arr)
            arr[y,x] = coord
        else:
            arr[y-1,x] = coord
    if z == 1:
        if at_edge(y,x,z, piece_num):
            arr = move_left(arr)
            arr[y,x] = coord
        else:
            arr[y,x+1] = coord
    if z == 2:
        if at_edge(y,x,z, piece_num):
            arr = move_up(arr)
            arr[y,x] = coord
        else:
            arr[y+1,x] = coord
    if z == 3:
        if at_edge(y,x,z,piece_num):
            arr = move_right(arr)
            arr[y,x] = coord
        else:
            arr[y,x-1] = coord
    return arr

def at_edge(y,x,dir, piece_num):
    if 0 > (x or y) or (x or y) > piece_num:
        print('Invalid inputs')
        return -1
    if dir == 0:
        return y-1 < 0
    if dir == 2:
        return y+1 > piece_num
    if dir == 1:
        return x+1 >= piece_num
    if dir == 3:
        return x-1 < 0

def move_down(arr):
    return np.roll(arr,1, axis=0)

def move_up(arr):
    return np.roll(arr,-1, axis=0)

def move_right(arr):
    return np.roll(arr,1, axis=1)

def move_left(arr):
    return np.roll(arr,-1, axis=1)

def compatibility(np_pieces):
    shape_results = (len(np_pieces),len(np_pieces[0]),4,len(np_pieces),len(np_pieces[0]))
    results = np.empty(shape_results)
    results.fill(np.nan)
    for x in range(0, len(np_pieces)):
        for y in range(0,len(np_pieces[x])):
            for z in range(4):
                for xx in range(0, len(np_pieces)):
                    for yy in range(0, len(np_pieces[x])):
                        if x != xx or y != yy:
                            zz = getInverse(z)
                            results[x][y][z][xx][yy] = get_score(np_pieces[x][y][z],np_pieces[xx][yy][zz],z)
            print('Piece ',x,y, ' completed...')
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

def print_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=600)
    nx.draw_networkx_edges(graph, pos, edge_color='black', width=3)
    labels = nx.get_edge_attributes(graph, 'p1')
    nx.draw_networkx_labels(graph, pos, font_size=20, font_family='sans-serif')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.axis('off')
    plt.show()


def find_best_start(graph):
    start_node = (n for n in graph if len(list(graph.neighbors(n))) == 1)
    cur_length = 10000
    for node in list(start_node):
        new_len = len(list(nx.single_source_shortest_path_length(graph, node, cutoff=3)))
        if new_len < cur_length:
            cur_length = new_len
            target_node = node
    return target_node


#check if move is possible up down left or right (none?) if so do it else move n to left right top or bottom and insert nbr in place of n
def reconstruct(graph, images):
    node_num = len(nx.nodes(graph))
    arr = np.zeros((node_num,node_num))
    edge_done = []
    start_node_tup = find_best_start(graph)
    tt = list(nx.bfs_successors(graph, start_node_tup))
    for node_list in tt:
        curr_node = node_list[0]
        y,x = curr_node
        if len(edge_done) == 0:
            arr[0,0] = convert_coord(y,x,node_num)
            edge_done.append(curr_node)
        res = np.transpose(np.where(arr == convert_coord(y,x,node_num)))
        y = res[0,0]
        x = res[0,1]
        for nodes in node_list:
            if nodes != curr_node:
                for node in nodes:
                    target_node = node
                    yy,xx = target_node
                    p1_data = graph[curr_node][target_node]['p1']
                    p2_data = graph[curr_node][target_node]['p2']
                    if p1_data[:2] == curr_node:
                        edge = p1_data[-1]
                    else:
                        edge = p2_data[-1]
                    arr = insert_piece(y,x,edge,arr,yy,xx)
                    edge_done.append(target_node)
    return arr

def display_image(arr,num_piece, images):
    arr = clean_results(arr)
    print('result array ',arr)
    arr_coord = [[None for _ in range(num_piece)] for _ in range(num_piece)]
    for y in range(len(arr)):
        for x in range(len(arr[y])):
            yy,xx = decode_coord(arr[y,x],num_piece)
            arr_coord[y][x] = (yy,xx)
    print('result array with coordinates adjusted ' ,arr_coord)
    return arr_coord

def clean_results(arr):
    arr = arr[~np.all(arr == 0, axis=1)]
    arr = np.transpose(arr)
    arr = arr[~np.all(arr == 0, axis=1)]
    arr = np.transpose(arr)
    return arr


def graph_create(results,images):
    graph = nx.Graph()
    for x in range(0, len(results)):
        for y in range(0, len(results[x])):
            for z in range(4):
                match_x, match_y, match_score = get_best_match(x,y,results,z)
                # Check if edge already exists, if it does only replace current edge if match_score is smaller than existing score (this can happen at edges)
                if not(graph.get_edge_data((x,y),(match_x,match_y)) is not None and graph.get_edge_data((x,y),(match_x,match_y))['weight'] > match_score):
                    graph.add_edge((x,y), (match_x,match_y), weight=round(match_score,5), p1=(x,y,z), p2=(match_x,match_y,getInverse(z)),object=images[x][y])
    T = nx.minimum_spanning_tree(graph)
    return T, graph
