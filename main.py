from splitter import pieces, edges, compatibility, graph_create, print_graph, get_best_match, is_best_match, reconstruct, display_image
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
height_og = len(img)
width_og = len(img[0])

def show_pieces(np_array):
    x= 0
    y = 0
    result = Image.new('RGB', (width_og,height_og))
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

def show_pieces_end(coord_arr,images):
    x= 0
    y = 0
    result = Image.new('RGB', (width_og,height_og))
    for i in range(0,len(coord_arr)):
        x=0
        for j in range(0,len(coord_arr[i])):
            xx = coord_arr[i][j][1]
            yy  = coord_arr[i][j][0]
            temp_img = Image.fromarray(images[yy][xx])
            x_offset = len(images[yy][xx][0,:])
            y_offset = len(images[yy][xx][:,0])
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

piece_number = 3
np_image = pieces(img,3,3)
result_edge = edges(np_image,4)
results = compatibility(result_edge)
graphres, graph_not_opt = graph_create(results,np_image)
res_arr_pos = reconstruct(graphres,np_image)
show_pieces(np_image)
#print_graph(graph_not_opt)
#print_graph(graphres)
display_arr = display_image(res_arr_pos, piece_number, np_image)
show_pieces_end(display_arr,np_image)
