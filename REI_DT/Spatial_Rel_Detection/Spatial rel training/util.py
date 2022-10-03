from gensim.models import KeyedVectors
import json
import pickle
import numpy as np
from autocorrect import spell
import cv2
import os
import random
from collections import defaultdict
import torch
import math


if not 'NO_WORD2VEC' in os.environ:
    print(' => loading the word2vec model..')
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, unicode_errors='ignore')
else:
    word2vec = defaultdict(lambda : np.zeros((300,), dtype=np.float32))
    print('WARNING: WORD2VEC IS NOT LOADED!')


def phrase2vec(phrase, max_phrase_len, word_embedding_dim):
    vec = np.zeros((max_phrase_len, word_embedding_dim,), dtype=np.float32)
    for i, word in enumerate(phrase.split()):
        assert i < max_phrase_len
        if word in word2vec:
            vec[i] = word2vec[word]
        elif spell(word) in word2vec:
            vec[i] = word2vec[spell(word)]
        else:
            pass
            #print(word)
    return vec


def onehot(k, n):
    encoding = np.zeros((n,), dtype=np.float32)
    encoding[k] = 1.
    return encoding


def read_img(url, imagepath):
    if url.startswith('http'):  # flickr
        filename = os.path.join(imagepath, 'flickr', url.split('/')[-1])
    else:  # nyu
        filename = os.path.join(imagepath, 'nyu', url.split('/')[-1])
    img = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
    assert img.shape[2] == 3
    return img

def multispafea(subbox,objbox):
    subbox = [subbox[2], subbox[0],subbox[3],subbox[1]]
    objbox = [objbox[2], objbox[0],objbox[3],objbox[1]]
    xmin1, ymin1, xmax1, ymax1 = subbox
    xmin2, ymin2, xmax2, ymax2 = objbox
    x1 = (xmax1 + xmin1)/2
    x2 = (xmax2 + xmin2)/2
    y1 = (ymax1 + ymin1)/2
    y2 = (ymax2 + ymin2)/2  
    w1 = (xmax1 - xmin1)
    h1 = (ymax1 - ymin1)

    dw1 = math.log(abs((x1-x2)/w1)) if abs((x1-x2)/w1) > 1 else 0
    dh1 = math.log(abs((y1-y2)/h1)) if abs((y1-y2)/h1) > 1 else 0
    w12 = math.log((xmax1- xmin1) / (xmax2- xmin2))
    h12 = math.log((ymax1- ymin1) / (ymax2- ymin2))
    # pos = [w1, h1, w12, h12]
    # position_encoding = np.array([
    #       [pos / np.pow(10000, 2.0 * (j // 2) / 64) for j in range(64)]
        
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    union_area = (area1+area2-inter_area+1e-6)
    # print(area1/area2, w12, h12, inter_area/area1, inter_area/area2, inter_area/union_area)
    return  dw1, dh1, w12, h12, area1/area2, inter_area/area1, inter_area/area2, inter_area/union_area

def accuracies(pred_file, gt_file, split, vis, args):
    gt = {}
    data = json.load(open(gt_file))
    for img in data:
        if img['split'] != split:
            continue
        for annot in img['annotations']:
            annot['url'] = img['url']
            gt[annot['_id']] = annot

    cnts = defaultdict(lambda : {'correct': 0, 'incorrect': 0})
    _ids, predictions = pickle.load(open(pred_file, 'rb'))
    for _id, prediction in zip(_ids, predictions):
        predicate = gt[_id]['predicate']
        if (prediction > 0.) == gt[_id]['label']:
            cnts[predicate]['correct'] += 1
            cnts['overall']['correct'] += 1
        else:
            cnts[predicate]['incorrect'] += 1
            cnts['overall']['incorrect'] += 1

    accs = {}
    for k, v in cnts.items():
        accs[k] = 100. * v['correct'] / (v['correct'] + v['incorrect'])
    return accs



