
######## bbox = [ymin, ymax, xmin, xmax]  ##########


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import numpy as np
import math
import statistics

sets=['train', 'test','val']


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2
    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = -statistics.median(y_vect)
        z_median = statistics.median(z_vect)
        thd=-0.60
        x_median1 = x_median
        y_median1 = math.cos(thd) * y_median + math.sin(thd) * z_median
        z_median1 = math.cos(thd) * z_median - math.sin(thd) * y_median
    except Exception:
        x_median1 = -1
        y_median1 = -1
        z_median1 = -1
        pass

    return x_median1, -y_median1, -z_median1

def convert_annotation(image_id):
    in_file_label = open('/home/lx/DRNet/datasets/data_yolo/Annotations/%s.xml'%( image_id))
    in_file_depth = np.load('/home/lx/DRNet/datasets/ptcloud/%s.npy'%(image_id))
    out_file = open('/home/lx/DRNet/datasets/dataToAnno/annotations_txt_60/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file_label)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(difficult)==1:
            continue
        # cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bbox = (float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text), float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text))
        x_extend = (bbox[3] - bbox[2])/2
        y_extend = (bbox[1] - bbox[0])/2
        bounds = [bbox[2] + x_extend,bbox[0] + y_extend,x_extend,y_extend]
        x, y, z = get_object_depth(in_file_depth, bounds)
        box_3d = np.array([x,y,z])
        box_3d = np.around(box_3d,decimals=2)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.write(str(cls) + " " + " ".join([str(a) for a in bbox]) + " " + str(box_3d[0]) + " " + str(box_3d[1]) +  " " + str(box_3d[2]) +'\n')


def main():
    for image_set in sets:
        # if not os.path.exists('/home/lx/datasets/dataToAnno/annotations_txt/'):
        #     os.makedirs('/home/lx/datasets/dataToAnno/annotations_txt/')
        image_ids = open('/home/lx/DRNet/datasets/data_yolo/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
        for image_id in image_ids:
            convert_annotation(image_id)
    
    

if __name__ == '__main__':
    main()




