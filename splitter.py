import numpy as np
from numpy import all
import matplotlib.pylab as plt
from PIL import Image
import matplotlib as plt
import cv2
import scipy.spatial.distance as SSD
import math

def pieces(image, target) :
    images = []
    np_image = []
    start_x = 0
    start_y = 0
    width, height = image.size
    split_height = height / target
    split_width = width / target
    for x in range(0, target):
        for y in range(0, target):
            end_x = start_x + split_width
            end_y = start_y + split_height
            box = (start_x, start_y, end_x, end_y)
            temp_img = image.crop(box)
            images.append(temp_img)
            start_y = start_y + split_height + 1
        start_x = start_x + split_width + 1
        start_y = 0
    temp_list = []
    for x in range(0, len(images)):
        temp_arr = np.array(images[x])
        temp_list.append(temp_arr)
    np_image = np.array(temp_list)
    #np.random.shuffle(np_image)
    return np_image



def edges(np_image) :
    h, w, c = np_image[0].shape
    edge_pieces = [[] for i in range(len(np_image))]
    target_size = 2
    north_box = (0, 0, w, target_size)
    south_box = (0, h-target_size,w, h)
    west_box = (0, 0, target_size, h)
    east_box = (w - target_size, 0, w, h)
    for x in range(0, len(np_image)) :
        north = Image.fromarray(np_image[x])
        north = north.crop(north_box)
        north = np.array(north)
        east = Image.fromarray(np_image[x])
        east = east.crop(east_box)
        east = np.array(east)
        west = Image.fromarray(np_image[x])
        west = west.crop(west_box)
        west = np.array(west)
        south = Image.fromarray(np_image[x])
        south = south.crop(south_box)
        south = np.array(south)
        edge_pieces[x].append(north)
        edge_pieces[x].append(east)
        edge_pieces[x].append(south)
        edge_pieces[x].append(west)
    np_edge = np.array(edge_pieces)
    return np_edge


def compatibility(np_pieces):
    results = []
    for x in range(0, len(np_pieces)):
        for y in range(0, len(np_pieces[x])):
                results.append(northsouth(np_pieces[x][y], np_pieces, x, y))
    for x, elem in enumerate(results):
        results[x].sort()
    return results

def print_results(results):
    piece_num = 0
    edge_num = -1
    for x, result in enumerate(results):
        edge_num += 1
        if x % 4 == 0 and x != 0:
            piece_num += 1
            edge_num = 0
        print('Piece number ' + str(piece_num) + ' edge number ' + str(edge_num))
        print(np.array(result))


def northsouth(target, pieces, xx, yy):
    target = plt.colors.rgb_to_hsv(target)
    ax = 0
    results = []
    if yy % 2 == 0:
        start = 0
    else:
        start = 1
    for x in range(0, len(pieces)):
     for y in range(start, len(pieces[x]), 2):
         #print(str(x) + ' ' + str(y))
         if (x != xx or y != yy):
            edgeCand = pieces[x][y]
            edgeCand = plt.colors.rgb_to_hsv(edgeCand)
            result_1 = mahalanobis(np.array(np.gradient(target[:,:,0], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,0], axis=ax)))
            result_1inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 0], axis=ax)),
                                      np.array(np.gradient(target[:, :, 0], axis=ax)))
            #result_1 = result_1 + result_1inv
            #result_1 = result_1.mean()
            result_1 = np.median(result_1)
            result_2 = mahalanobis(np.array(np.gradient(target[:,:,1], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,1], axis=ax)))
            result_2inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 1], axis=ax)),
                                      np.array(np.gradient(target[:, :, 1], axis=ax)))
            #result_2 = result_2 + result_2inv
            result_2 = np.median(result_2)
            #result_2 = result_2.mean()
            result_3 = mahalanobis(np.array(np.gradient(target[:,:,2], axis=ax)),
                                   np.array(np.gradient(edgeCand[:,:,2], axis=ax)))
            result_3inv = mahalanobis(np.array(np.gradient(edgeCand[:, :, 2], axis=ax)),
                                      np.array(np.gradient(target[:, :, 2], axis=ax)))
            #result_3 = result_3 + result_3inv
            result_3 = np.median(result_3)
            #result_3 = result_3.mean()
            result = result_1 +result_2+result_3
            results.append([result,x, y])
    return results


def mahalanobis(colchan1, colchan2):
    M = np.array([colchan1.mean(), colchan2.mean()])
    covariance = np.cov([colchan1.ravel(), colchan2.ravel()])
    try:
        inv_cov = np.linalg.inv(covariance)
    except Exception:
        inv_cov = np.array([[0,1],[1,0]])

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
        score, piece, position = results[x][0]
        target_image = np_pieces[piece]
        target_image = Image.fromarray(target_image)
        if position == 3 and piece == 2:
            result.paste(target_image, (w, 0))
        if position == 0 and piece == 1:
            result.paste(target_image, (0, h))
        if position == 0and piece == 3:
            result.paste(target_image, (w, h))
    result.show()