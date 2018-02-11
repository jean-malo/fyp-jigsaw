import numpy as np
from numpy import all
import matplotlib.pylab as plt
from PIL import Image
import matplotlib as plt
import cv2
import scipy.spatial.distance as SSD

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
    target_size = 4
    north_box = (0, 0, w, target_size)
    south_box = (0, h-target_size,w, h)
    for x in range(0, len(np_image)) :
        north = Image.fromarray(np_image[x])
        north = north.crop(north_box)
        north = np.array(north)
        south = Image.fromarray(np_image[x])
        south = south.crop(south_box)
        south = np.array(south)
        edge_pieces[x].append(north)
        edge_pieces[x].append(south)
    np_edge = np.array(edge_pieces)
    return np_edge


# The gradient is computed using second order accurate central differences
# in the interior and either first differences or second order accurate one-sides
# (forward or backwards) differences at the boundaries.
# The returned gradient hence has the same shape as the input array.
def edge_gradient(edge1, edge2) :
    edge = np.append(edge1, edge2, axis=1)
    edge_hsv = plt.colors.rgb_to_hsv(edge)
    current_score = getGradientMagnitude(edge_hsv)
    return current_score

def getScores(pieces):
    links = {}
    results= []
    temp_results = []
    for x, piece in enumerate(pieces):
        for y, edge in enumerate(piece):
            target_x, target_y = northsouth(edge, pieces)
            temp_results.append([[x,y], [target_x, target_y]])
            results.append(temp_results)
    return temp_results

def compatibility(np_pieces):
    results = []
    for x in range(0, len(np_pieces)):
        for y in range(0, len(np_pieces[x])):
            results.append(northsouth(np_pieces[x][y], np_pieces, x, y))
    results = np.array(results)
    return results

def northsouth(target, pieces, xx, yy):
    results = []
    for x in range(0, len(pieces)):
     for y in range(0, len(pieces[x])):
         if (x != xx and y != yy):
            edgeCand = pieces[x][y]
            result_1 = mahalanobis(np.array(np.gradient(target[:,:,0], axis=-1)), np.array(np.gradient(edgeCand[:,:,0], axis=-1)))
            result_2 = mahalanobis(target[:,:,1], edgeCand[:,:,1])
            result_3 = mahalanobis(target[:,:,2], edgeCand[:,:,2])
            results.append([result_1,result_2, result_3])
    return results


def mahalanobis(colchan1, colchan2):
    M = np.array([colchan1.mean(), colchan2.mean()])
    covariance = np.cov([colchan1.ravel(), colchan2.ravel()])
    inv_cov = np.linalg.inv(covariance)
    D = np.dstack([colchan1, colchan2]).reshape(-1, 2)
    result = SSD.cdist(D, M[None, :], metric='mahalanobis', VI=inv_cov)
    result = result.reshape(colchan1.shape)
    return result
