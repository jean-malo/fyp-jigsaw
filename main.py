from splitter import pieces, edges, compatibility, northsouth, print_results, reconstruct
from PIL import Image
import math
import numpy as np




def show_pieces(np_array):
    #count = np_array[x]
    total = np_array.shape[0]
    h = np_array[0].shape[0]
    w = np_array[0].shape[1]
    target_width = math.sqrt(total) * w
    target_height = math.sqrt(total) * h
    x = 0
    y = 0
    result = Image.new('RGB', (int(target_width), int(target_height)))
    for image in np_array :
        #count += 1
        if x >= target_width :
            x = 0
            y+=image.shape[0]
        img = Image.fromarray(image)
        result.paste(img, (x,y))
        x += image.shape[1]
    result.show()


img = Image.open('pacman.jpg')
np_image = pieces(img, 2)
result_edge = edges(np_image)
results = compatibility(result_edge)
print(results)
print_results(results)
show_pieces(np_image)
#reconstruct(results, np_image)
