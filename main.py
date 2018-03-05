from splitter import pieces, edges, compatibility, graph_create, print_graph, get_best_match, is_best_match, reconstruct
from PIL import Image
import math
import scipy.spatial.distance as SSD

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv
from random import shuffle
np.set_printoptions(suppress=True)

def show_pieces(np_array):
    x= 0
    y = 0
    result = Image.new('RGB', (1000,668))
    print(len(np_array))
    for i in range(0,len(np_array)):
        x=0
        for j in range(0,len(np_array[i])):
            temp_img = Image.fromarray(np_array[i][j])
            x_offset = len(np_array[i][j][0,:])
            y_offset = len(np_array[i][j][:,0])
            result.paste(temp_img, (x,y))
            x += x_offset
        y += y_offset
    result.show()
    return result

#Shuffles a 2d array by turning the array in a 1d array and then back to 2d
def shuflle2dArray(np_array):
    np_array = np.array(np_array)
    arr_shape = np_array.shape
    new_arr = np_array.reshape(arr_shape[0]*arr_shape[1],arr_shape[-3],arr_shape[-2],arr_shape[-1])
    np.random.shuffle(new_arr)
    arr = new_arr.reshape(arr_shape)
    np_array = arr
    return np_array


def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape
  (width, height, channels)
  """
  with Image.open(image_path) as image:
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))
  return im_arr

img = jpg_image_to_array('fez.jpg')
np_image = pieces(img,5,5)
show_pieces(np_image)
result_edge = edges(np_image,4)
results = compatibility(result_edge)
graphres = graph_create(results)
print(is_best_match(0,0,0,1,results,3))
res_place =reconstruct(graphres, results)
print(res_place[0][0][1])
print_graph(graphres)
