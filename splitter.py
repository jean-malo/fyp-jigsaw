import numpy as np
from PIL import Image
import scipy.spatial.distance as SSD
import math
import networkx as nx
import cv2 as cv
import matplotlib as plt
plt.use('TkAgg')
import matplotlib.pyplot as plt2


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
    return images

def edges(np_image) :
    #h, w, c = np_image[0].shape
    edge_pieces = []
    target_size = 5
    #north_box = (0, 0, w, target_size)
    #south_box = (0, h-target_size,w, h)
    #west_box = (0, 0, target_size, h)
    #east_box = (w - target_size, 0, w, h)
    for x in range(0, len(np_image)) :
        for y in range(0,len(np_image[x])):
            north = np_image[x][y][0:target_size,:,:]
            south = np_image[x][y][-target_size-1:,:,:]
            west = np_image[x][y][:, 0:target_size, :]
            east = np_image[x][y][:, -target_size-1:, :]
            Image.fromarray(north).show()
            Image.fromarray(south).show()
            Image.fromarray(south).show()
            Image.fromarray(south).show()
    np_edge = np.array(edge_pieces)
    #return np_edge


def compatibility(np_pieces):
    results = []
    for x in range(0, len(np_pieces)):
        results.append(northsouth(np_pieces[x], np_pieces, x))
    for x, elem in enumerate(results):
        results[x].sort()
    return results

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

def northsouth(target, pieces, xx):
    target = plt.colors.rgb_to_hsv(target)
    ax = 0
    results = []
    if xx % 2 == 0:
        start = 0
    else:
        start = 1
    for x in range(start, len(pieces), 2):
         #print(str(x) + ' ' + str(y))
         if (x != xx):
            edgeCand = pieces[x]
            edgeCand = plt.colors.rgb_to_hsv(edgeCand)
            result_1 = mahalanobis(np.array(np.gradient(target[:,:,0], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,0], axis=ax)))
            result_1inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 0], axis=ax)),
                                      np.array(np.gradient(target[:, :, 0], axis=ax)))
            result_1 = np.median(result_1)
            result_2 = mahalanobis(np.array(np.gradient(target[:,:,1], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,1], axis=ax)))
            result_2inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 1], axis=ax)),
                                      np.array(np.gradient(target[:, :, 1], axis=ax)))
            result_2 = np.median(result_2)
            result_3 = mahalanobis(np.array(np.gradient(target[:,:,2], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,2], axis=ax)))
            result_3inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 2], axis=ax)),
                                      np.array(np.gradient(target[:, :, 2], axis=ax)))
            result_3 = np.median(result_3)
            result = result_1 +result_2+result_3
            results.append([result,x])
    return results


def mahalanobis(colchan1, colchan2):
    epsilon = 10**-6
    dummy_gradients = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1],[0,0,0]]
    #colchan1= np.append(colchan1, dummy_gradients)
    #colchan2 = np.append(colchan2, dummy_gradients)
    M = np.array([colchan1.mean(), colchan2.mean()])
    covariance = np.cov([colchan1.ravel(), colchan2.ravel()])
    try:
        inv_cov = np.linalg.inv(covariance)
    except Exception:
        inv_cov = np.linalg.inv(covariance + np.eye(covariance.shape[1]) * epsilon)
    D = np.dstack([colchan1, colchan2]).reshape(-1, 2)
    result = SSD.cdist(D, M[None, :], metric='mahalanobis', VI=inv_cov)
    result = result.reshape(colchan1.shape)
    return result


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
    np_results = np.array(results)
    result_clean = []
    #print(np.array(results))
    np_results = np_results[np.logical_not(np.logical_or(np.isnan(np_results[:, :, 0]),
                                                         np.less(np_results[:, :, 0], 10 ** -6)))]
    print(np.reshape(np_results))
    #for x, result in enumerate(np_results):
        #result[0][0] = result[0][0] / result[1][0]
        #result_clean.append(result[0])
    #print(result_clean)
    return result_clean

def graph_init(results):
    #print(np.array(results))
    graphs = []
    nodes = [x for x in range(0,len(results))]
    count = 0
    piece_num = 0
    graph = nx.Graph()
    start = 0
    end = 4
    for y in range(0, len(results) // 4):
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
        plt2.axis('off')
        plt2.show()
