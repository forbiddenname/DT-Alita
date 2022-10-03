#!python3
"""
online relation detector and physical space transmitter

@author: lixin, TongjiUni

@date: 20200808
"""
# pylint: disable=R, W0401, W0614, W0703
import os
import sys
import time
import logging
import random
from random import randint
import math
import statistics
import getopt
from ctypes import *
import numpy as np
import cv2
import pyzed.sl as sl

import torch 
import torchvision.transforms as transforms
from models.recurrent_phrase_encoder import RecurrentPhraseEncoder
from models.drnet_depth1 import DRNet_depth
from util_depth import phrase2vec
from options1 import parse_args
import select
import socket
import queue
from time import sleep
import threading
import json
from datetime import datetime



predicate_categories = [   'nothing',
                                'above',
                                'behind',
                                'front',
                                'below',
                                'left',
                                'right',
                                'on',
                                'under',
                                'in',
                                'with',
                                'hold' 
                                ]

object_categories = ['manipulator',
                     'sponge',
                     'cup',
                     'bowl',
                     'plate',
                     'apple',
                     'lid',
                     'spoon',
                     'sandwich',
                     'banana',
                     'orange',
                     'ball'
                    ]

relprelist = []  
force = [] 
stiff = [] 
byteimage = 0 

####boundingbox和img 预处理并转tensor
#region
t_bbox = transforms.Compose([
transforms.ToPILImage('RGB'),
transforms.Pad(4, padding_mode='edge'),
transforms.CenterCrop(32),   
transforms.ToTensor(),
])       

t_bboximg = transforms.Compose([
transforms.ToPILImage('RGB'),
transforms.CenterCrop(224),
transforms.ToTensor(),
])   
#endregion

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def getBBox(bounds):
    ### (x,y,w,h) -----> (ymin, ymax, xmin, xmax)
    y_extent = int(bounds[3])
    x_extent = int(bounds[2])
    x_coord = int(bounds[0] - bounds[2]/2)
    y_coord = int(bounds[1] - bounds[3]/2)
    return [y_coord, y_coord + y_extent, x_coord, x_coord + x_extent]

def getDualMask(ih, iw, bb, heatmap_size=32):  #求dual spatial masks
    rh = float(heatmap_size) / ih
    rw = float(heatmap_size) / iw
    x1 = max(0, int(math.floor(bb[0] * rh)))
    x2 = min(heatmap_size, int(math.ceil(bb[1] * rh)))
    y1 = max(0, int(math.floor(bb[2] * rw)))
    y2 = min(heatmap_size, int(math.ceil(bb[3] * rw)))
    mask = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    mask[x1 : x2, y1 : y2] = 255
    #assert(mask.sum() == (y2 - y1) * (x2 - x1))
    return mask

def spatial_fea(subb, objb, ih, iw):
    bbox_mask = np.stack([getDualMask(ih, iw, subb, 32).astype(np.uint8), 
                        getDualMask(ih, iw, objb, 32).astype(np.uint8), 
                        np.zeros((32, 32), dtype=np.uint8)], 2)
    bbox_mask = t_bbox(bbox_mask)[:2].float() / 255.
    bbox_mask = torch.unsqueeze(bbox_mask, dim=0).float()
    return bbox_mask

def enlarge(bbox, factor, ih, iw):
    height = bbox[1] - bbox[0]
    width = bbox[3] - bbox[2]
    assert height > 0 and width > 0
    return [max(0, int(bbox[0] - (factor - 1.) * height / 2.)),
            min(ih, int(bbox[1] + (factor - 1.) * height / 2.)),
            max(0, int(bbox[2] - (factor - 1.) * width / 2.)),
            min(iw, int(bbox[3] + (factor - 1.) * width / 2.))]

def getUnionBBox(aBB, bBB, ih, iw, margin = 10):
    return [max(0, min(aBB[0], bBB[0]) - margin), \
            min(ih, max(aBB[1], bBB[1]) + margin), \
            max(0, min(aBB[2], bBB[2]) - margin), \
            min(iw, max(aBB[3], bBB[3]) + margin)]

def getAppr(im, bb,out_size=224.):
        subim = im[bb[0] : bb[1], bb[2] : bb[3], :]

        subim = cv2.resize(subim, None, None, out_size / subim.shape[1], out_size / subim.shape[0], interpolation=cv2.INTER_LINEAR)

        subim = subim[:,:,0:3]   
        subim = (subim / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return subim.astype(np.float32, copy=False)

def img_fea(subb,objb,ih,iw,factor,img):
    union_bbox = enlarge(getUnionBBox(subb, objb, ih, iw), factor, ih, iw)
#     img = np.expand_dims(img,0)
    bbox_img = t_bboximg(getAppr(img, union_bbox))
    bbox_img = torch.unsqueeze(bbox_img, dim=0).float()
    return bbox_img

# def semRea(stiffnow,count):



def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#region
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                log.info("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # log.info(os.environ.keys())
            # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            log.warning("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            log.warning("Environment variables indicated a CPU run, but we didn't find `" +
                        winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("/home/lx/DRNet/detection_yolo/libdarknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int
predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)
if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]
make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE
get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)
make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)
free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]
free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]
network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]
reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]
load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p
load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p
do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
free_image = lib.free_image
free_image.argtypes = [IMAGE]
letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE
load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA
load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE
rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]
predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)
#endregion

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, depth, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    # dets_type : <class '__main__.LP_DETECTION'>

    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        log.debug("about to range")
    for j in range(num):
        for i in range(meta.classes):
            # if dets[j].prob[i] > 0 and i in [41,46,47,48]:
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                bounds = (b.x, b.y, b.w, b.h)
                # x, y, z = get_object_depth(depth, bounds)
                box_3d = np.array(get_object_depth(depth, bounds),np.float32)
                # box_3d=np.array([x, y, z])
                if j!= 0 and name_tag == 'manipulator':  
                   res.append(res[0])
                   res[0] = (name_tag, dets[j].prob[i], np.around(bounds), i, np.around(box_3d, decimals = 2))
                else:
                   res.append((name_tag, dets[j].prob[i], np.around(bounds), i, np.around(box_3d, decimals = 2)))
                # res.append((name_tag, dets[j].prob[i], np.around(bounds), i, np.around(box_3d, decimals = 2)))      #####object label , confidence , boundingbox , i , depthdata
    # res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

netMain = None
metaMain = None
altNames = None

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
        thd=-0.65   ###-0.60/-0.65
        x_median1 = x_median
        y_median1 = math.cos(thd) * y_median + math.sin(thd) * z_median
        z_median1 = math.cos(thd) * z_median - math.sin(thd) * y_median
    except Exception:
        x_median1 = -1
        y_median1 = -1
        z_median1 = -1
        pass


    return x_median1, y_median1, z_median1

def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array

def databytetolist(robot_data, rel_data,force_mea1):
    robot_datastr = robot_data.decode()
    robot_datastr = robot_datastr.replace('p','').replace('[','')
    robot_datalist = robot_datastr.split(']')
    datalist = []
    for idx, data in enumerate(robot_datalist):
        a = []
        for idx1,data1 in enumerate(data.split(',')):
            a.append(round(float(data1),2))
            # a.append(float(data1))
        datalist.append(a)
    datalist+= [[force_mea1]]    
    datalist+= rel_data          
    return datalist               

def databytetolist1(robot_data, force_mea1):
    robot_datastr = robot_data.decode()
    robot_datastr = robot_datastr.replace('p','').replace('[','')
    robot_datalist = robot_datastr.split(']')
    datalist = []
    for idx, data in enumerate(robot_datalist):
        a = []
        for idx1,data1 in enumerate(data.split(',')):
            a.append(round(float(data1),2))
            # a.append(float(data1))
        datalist.append(a)
    datalist+= [[force_mea1]]            
    return datalist              


def datasyn1(datalist,data_listpre,count):
    ##### Todo updating strategy #####
    c = datalist
    p = data_listpre
    cf = c[4][0]
    pf = p[4][0]
    datatounity = [[0] for i in range(len(c))]
    e = 1 ###
    T = 0.033 ###频率 100Hz
    if c != p:
        if abs(cf) < e: ####力传感器的精度   ###接触?
            datatounity[0] = c[0]
            if (len(c) >5 ):                        
                if (len(p)>5 and c[5:] == p[5:]):        
                    datatounity[5:] = [[0]]             
                else:
                    datatounity[5:] = c[5:]
            print('approach')
        elif abs((cf - pf)/ T) < e:    ###力无变化?
            datatounity[0] = c[0]
 
            if (len(c) >5 ):                                
                if (len(p)>5 and c[5:] == p[5:]):        
                    datatounity[5:] = [[0]]             
                else:
                    datatounity[5:] = c[5:]
            print('move together')
        elif (cf - pf)/ T > e:

            datatounity[3] = c[3]
            datatounity[4] = c[4]
            print('grasp')
        else:
            datatounity[3] = c[3]

    datatounity[1:3] = c[1:3] 
    return datatounity     

##threading 1, data to digital twin
def toUnity():
    global relprelist
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(False)
    server_address = ('192.168.1.115', 30004)
    print ('starting up on %s port %s' % server_address)
    server.bind(server_address)
    server.listen(5)
    inputs = [server]
    outputs = []
    data_listpre = []
    data_listpre1 = []
    time  = datetime.strftime(datetime.now(),'%m%d%H%M')
    # datasave =  open("/home/lx/DRNet/experiment/synchronization_strategy/"+time+".txt", "a")
    while inputs:
        # sleep(0.02)    ###50Hz
        readable, writable, exceptional = select.select(inputs, outputs, inputs)
        mark1 = 0  ##count
        for s in readable:
            if s is server:
                connection, client_address = s.accept()
                print ('connection from', client_address)
                # connection.setblocking(0)
                inputs.append(connection)
                outputs.append(connection)
            else:
                head_data=s.recv(3)
                if head_data != '':
                    # package_size = ord(head_data)   ### acill转十进制
                    package_size=int(head_data)

                    recv_data = s.recv(package_size)
                    
                    # semRea(stiffness1,mark1)
                    time2 = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
                    fh2 = open('/home/lx/DRNet/experiment/qualitative_evaluation/10_28_num2/rel'+time2+'_'+str(mark1)+'.txt', 'w', encoding='utf-8')
                    fh2.write(str(relprelist).replace('],',':').replace(']','').replace('[',' '))
                    fh2.close()


                    data_list = databytetolist(recv_data, relprelist, force_mea)

                    if len(data_listpre)!=0:
                        send_data = datasyn(data_list,data_listpre,mark1)

                    else:
                        send_data = data_list
                        
                    data_listpre = data_list

                    send_data = json.dumps(send_data)
                    
                    send_data_size = len(send_data)
                    
                    send_datasize= json.dumps(str(send_data_size//100%10)+str(send_data_size//10%10)+str(send_data_size//1%10))

                    mark1+=1
                    writable[0].send(send_datasize.encode())    #chr---转成ascill码
                    writable[0].send(send_data.encode())
                else:
                    print ('closing', client_address)
                    if s in outputs:
                        outputs.remove(s)
                    inputs.remove(s)
                    s.close()
                    datasave.close()
        for s in exceptional:
            print ('exception condition on', s.getpeername())
            # Stop listening for input on the connection
            inputs.remove(s)
            if s in outputs:
                outputs.remove(s)
    datasave.close()

##threading 2, rel prediction
def predrel(argv):
    
    ##################################3# yolo detection weight
    thresh = 0.25
    darknet_path="/home/lx/DRNet/detection_yolo/libdarknet/"
    # config_path = darknet_path + "cfg/yolov3.cfg"
    config_path = darknet_path + "cfg/yolov3-voc.cfg"
    # weight_path = "yolov3.weights"
    weight_path = "/home/lx/DRNet/datasets/data_yolo/backup/yolov3-voc_1300.weights"
    #weight_path = "yolov4.weights"
    meta_path = darknet_path + "cfg/yolo.data"
    # meta_path = "coco.data"
   
   ##################################3# spatial relation detection weight
    # checkpoint_path = "/home/lx/DRNet/SpatialSense/baselines/runs/drnet/checkpoints/model_datasetReSe_29.pth"
    checkpoint_path = "/home/lx/DRNet/SpatialSense/baselines/runs/customon_depthon_apprfeaton_spatialdualmask/checkpoints/model_96.970_28.pth"

    # args = parse_args()
    
    svo_path = None
    zed_id = 0

    # help_str = 'darknet_zed.py -c <config> -w <weight> -m <meta> -t <threshold> -s <svo_file> -z <zed_id>'
    # try:
    #     opts, args = getopt.getopt(
    #         argv, "hc:w:m:t:s:z:", ["config=", "weight=", "meta=", "threshold=", "svo_file=", "zed_id="])
    # except getopt.GetoptError:
    #     log.exception(help_str)
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         log.info(help_str)
    #         sys.exit()
    #     elif opt in ("-c", "--config"):
    #         config_path = arg
    #     elif opt in ("-w", "--weight"):
    #         weight_path = arg
    #     elif opt in ("-m", "--meta"):
    #         meta_path = arg
    #     elif opt in ("-t", "--threshold"):
    #         thresh = float(arg)
    #     elif opt in ("-s", "--svo_file"):
    #         svo_path = arg
    #     elif opt in ("-z", "--zed_id"):
    #         zed_id = int(arg)
    args = parse_args()
    input_type = sl.InputType()
    if svo_path is not None:
        log.info("SVO file : " + svo_path)
        input_type.set_from_svo_file(svo_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(zed_id)

    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()
    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    depth_img = sl.Mat()
    point_cloud_mat = sl.Mat()
    resl=sl.Resolution(640, 480)
    # Import the global variables. This lets us instance Darknet once,
    # then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In thon 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as meta_fh:
                meta_contents = meta_fh.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as names_fh:
                            names_list = names_fh.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass
    
    ###load relation detection weight
    phrase_encoder = RecurrentPhraseEncoder(300, 300)
    checkpoint = torch.load(checkpoint_path)
    model = DRNet_depth(phrase_encoder, 512, 3,args)
    model.cuda()
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    color_array = generate_color(meta_path)
    log.info("Running...")
    
    key = ''
    mark2 = 0
    while key != 113:  # for 'q' key
        start_time = time.time() # start time of the loop
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT,resolution=resl)
            image = mat.get_data()
            # print(image)
            ih, iw = image.shape[:2]
            # cam.retrieve_image(depth_img, sl.VIEW.DEPTH, resolution=resl)

            cam.retrieve_measure(
                point_cloud_mat, sl.MEASURE.XYZ,resolution=resl)
            depth = point_cloud_mat.get_data()
            
            # Do the detection
            detections = detect(netMain, metaMain, image, depth, thresh)     #####object label , confidence , boundingbox , i , depthdata
            time1 = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
            cv2.imwrite('/home/lx/DRNet/experiment/qualitative_evaluation/10_28_raw/image_nobbox'+time1+'_'+str(mark2)+'.jpg',image)
            # print(detections)
            # log.info(chr(27) + "[2J"+"**** " + str(len(detections)) + " Results ****")
            global relprelist
            relprelist = []
            relprelist1 = []
            ########### bits after image compression
            # global byteimage
            # img_encode = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 99])[1]
            # byteimage = img_encode.tostring()
            # print (len(byteimage))
            ########### bits after image compression
            detectnum = len(detections)
            #### batchsize data input
            mark = 0
            
            sub_obj_list = []
            for idx in range(0,detectnum):
                label_sub = detections[idx][0]
                sub1 = phrase2vec(label_sub,2,300)
                sub = torch.unsqueeze(torch.Tensor(np.array(sub1, np.float32)), dim=0)

                bounds_sub = detections[idx][2]
                box3d_sub1 = detections[idx][4]
                box3d_sub = torch.unsqueeze(torch.Tensor(box3d_sub1), dim=0)

                subbbox = getBBox(bounds_sub)

                for idx1 in range(idx+1, detectnum):
                    label_obj = detections[idx1][0]
                    obj1 = phrase2vec(label_obj,2,300)
                    obj = torch.unsqueeze(torch.Tensor(np.array(obj1, np.float32)), dim=0)

                    bounds_obj = detections[idx1][2]  
                    box3d_obj1 = detections[idx1][4]
                    box3d_obj = torch.unsqueeze(torch.Tensor(box3d_obj1), dim=0)

                    objbbox = getBBox(bounds_obj)
                    bbox_mask = spatial_fea(subbbox, objbbox, ih, iw)  ###bbox空间特征
                    bbox_img = img_fea(subbbox,objbbox,ih,iw,1.25,image) ###bbox图像特征

                    if mark < 1:
                       label_allsub = sub
                       label_allobj = obj
                       box3_allsub = box3d_sub
                       box3_allobj = box3d_obj
                       mask_allbbox = bbox_mask
                       img_allbbox = bbox_img

                    else:
                       label_allsub = torch.cat((label_allsub,sub),dim=0)
                       label_allobj = torch.cat((label_allobj,obj),dim=0)
                       img_allbbox  = torch.cat((img_allbbox,bbox_img),dim=0)
                       mask_allbbox = torch.cat((mask_allbbox,bbox_mask),dim=0)
                       box3_allsub  = torch.cat((box3_allsub,box3d_sub),dim=0)
                       box3_allobj  = torch.cat((box3_allobj,box3d_obj),dim=0)
                    
                    sub_obj_list.append([label_sub,label_obj])
                    mark+=1

                thickness = 1
                cv2.rectangle(image, (subbbox[2] - thickness, subbbox[0] - thickness),(subbbox[3] + thickness, subbbox[1] + thickness),color_array[detections[idx][3]], int(thickness*2))

            
            fh = open('/home/lx/DRNet/experiment/qualitative_evaluation/10_28_raw/rel'+time1+'_'+str(mark2)+'.txt', 'w', encoding='utf-8')
            fh1 = open('/home/lx/DRNet/experiment/qualitative_evaluation/10_28_num1/rel'+time1+'_'+str(mark2)+'.txt', 'w', encoding='utf-8')
            if mark > 0:
                relprelabel = model(label_allsub.cuda(), label_allobj.cuda(), img_allbbox.cuda(), mask_allbbox.cuda(),1,box3_allsub.cuda(),box3_allobj.cuda(),args)
                for idxx,label in enumerate(relprelabel):
                    rellabel_t = label.argmax()
                    sub_t = object_categories.index(sub_obj_list[idxx][0])
                    obj_t = object_categories.index(sub_obj_list[idxx][1])
                    relpair_t = [sub_t,rellabel_t.item(),obj_t]

                    rellabel = predicate_categories[rellabel_t]
                    relpair = [sub_obj_list[idxx][0], rellabel, sub_obj_list[idxx][1]]  
                    fh.write(str(relpair)+'\r\n')

                    if (not relpair_t in relprelist1):
                        fh1.write(str(relpair_t).replace('[','').replace(']','').replace(',',' ')+'\r\n')
                        relprelist1.append(relpair_t)

                relprelist = relprelist1  
                  
                
            cv2.imshow("ZED", image)
            cv2.imwrite('/home/lx/DRNet/experiment/qualitative_evaluation/10_28_raw/image'+time1+'_'+str(mark2)+'.jpg',image)
            fh.close()
            fh1.close()

            key = cv2.waitKey(5)
            mark2+=1

        else:
            key = cv2.waitKey(5) 
    cv2.destroyAllWindows()
    cam.close()
    log.info("\nFINISH")

def main(argv):
    
    t_rel = threading.Thread(target=predrel, args=(argv,))
    t_toUnity = threading.Thread(target=toUnity)
    t_rel.start()
    t_toUnity.start()
    with open('/home/lx/DRNet/experiment/synchronization_strategy/force.txt','r') as f:
        for line in f:
            force.append(line.strip('\n').replace(' ','').split(','))
    with open('/home/lx/DRNet/experiment/synchronization_strategy/stiffness.txt','r') as f:
        for line in f:
            stiff.append(line.strip('\n').replace(' ','').split(','))
if __name__ == "__main__":
    main(sys.argv[1:]) 
