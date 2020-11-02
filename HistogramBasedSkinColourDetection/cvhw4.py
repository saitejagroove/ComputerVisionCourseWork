from PIL import Image
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys

def readImage(image_path):
    """
    Function to read an image and return a sequence object which has the pixel
    """
    image = Image.open(image_path)
    image = image.getdata()
    return image

def getHistogram(image, hist_dict, color_type):
    """
    We get the histogram values using thos function
    """
    width, height = image.size
    for row in range(height):
        for col in range(width):
            r,g,b = image.getpixel((col, row))
            if color_type == 0:
                x,y,z = (math.floor(r/255.*10), math.floor(g/255.*10), math.floor(b/255.*10))
            elif color_type == 1:
                x,y,z = colorsys.rgb_to_hsv(math.floor(r/255.*10), math.floor(g/255.*10), math.floor(b/255.*10))
            else:
                s = r+g+b
                x = math.floor(r / float(s) * 100)
                y = math.floor(g / float(s) * 100)
                z = math.floor(b / float(s) * 100)
            if not (x,y) in hist_dict:
                hist_dict[(x,y)] = 1
            else:
                hist_dict[(x,y)] += 1
    return hist_dict

def train(image_paths, color_type):
    hist_dict = {}
    for image_path in image_paths:
        training_image = readImage(image_path)
        hsv_dict = getHistogram(training_image, hist_dict, color_type)
    totalValue = 0
    for value in hist_dict.values():
        totalValue += value
    for key in hist_dict.keys():
        hist_dict[key] /= float(totalValue)
    return hist_dict


def plot3D(hist):
    fig = plt.figure()
    num_elements = len(hist)
    ax1 = fig.add_subplot(111, projection='3d')
    xpos = []
    ypos = []
    dz = []
    for i in hist.keys():
        xpos.append(i[0])
        ypos.append(i[1])
        dz.append(hist[i])
    zpos = [0 for i in range(num_elements)]
    dx = [0.002 for i in range(num_elements)]
    dy = [0.01 for i in range(num_elements)]

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue')

def color_segmentation(image, threshold, color_type,trainingSize):
    image_paths = []
    for i in range(1, trainingSize):
        image_path = "trainingData/sample_" + str(i) + ".jpg"
        image_paths.append(image_path)
    hsv_dict = train(image_paths, color_type)
    # normalized_hsv_dict = normalize_histogram_sum(hsv_dict)
    plot3D(hsv_dict)
    width, height = image.size
    new_image = [[0 for x in range(width)] for x in range(height)]

    for row in range(height):
        for col in range(width):
            r,g,b = image.getpixel((col, row))
            if color_type == 0:
                x,y,z = (math.floor(r/255.*10), math.floor(g/255.*10), math.floor(b/255.*10))
            elif color_type == 1:
                x,y,z = colorsys.rgb_to_hsv(math.floor(r/255.*10), math.floor(g/255.*10), math.floor(b/255.*10))
            else:
                s = r+g+b
                x = math.floor(r / float(s) * 100)
                y = math.floor(g / float(s) * 100)
                z = math.floor(b / float(s) * 100)
            if (x,y) in hsv_dict and hsv_dict[(x,y)] > threshold:
                new_image[row][col] = (r,g,b)
            else:
                new_image[row][col] = (0,0,0)
    return new_image

def segmentedImage(image_array, name):
    """
    Function to save the segmented image
    """
    array = (np.array(image_array)).astype(np.uint8)
    image = Image.fromarray(array)
    # Final segmented image 
    image.save(name + ".bmp", "bmp")

if __name__ == "__main__":
    imagename = "gun1"
    threshold = 0.002
    color_type = 1
    #define trainingSize : num of images 2 - 17
    trainingSize = 16
    image_path = imagename +".bmp" 
    image = readImage(image_path)
    width, height = image.size
    image_array = color_segmentation(image, threshold, color_type,trainingSize)
    segmentedImage(image_array, "Results/segmented_" +imagename+str(trainingSize)+str(threshold))
    #plt.show()
    plt.savefig("Results/hist_"+imagename+str(trainingSize)+str(threshold)+".png")
