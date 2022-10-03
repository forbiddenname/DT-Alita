import json

predicate_categories = ['nothing',
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
                        'hold']
 
stat_num = {'nothing' : 0,
            'above' : 0,
            'behind' : 0,
            'front': 0,
            'below': 0,
            'left': 0,
            'right': 0,
            'on': 0,
            'under': 0,
            'in': 0,
            'with': 0,    
            'hold': 0}

def getanno(txtpath, img_idx):
    txt = open(txtpath + '/annotations_txt' + '/%s.txt'%(img_idx))
    txtD = []
    for line in txt:
        txtD.append([j for j in line.split()])
    
    pairlistindx = []
    for idx,data in enumerate(txtD):
        for idx1,data1 in enumerate(data[8::1]):
            if data1 != '0':
                pairlistindx.append([idx,idx1,int(data1)])
    pairlist = []

    for pair in pairlistindx:
        # print(pair)
        pairdic = {}
        pairdic['predicate'] = predicate_categories[pair[2]]

        for cate, num in stat_num.items():
            if (cate == pairdic['predicate']):
                stat_num[cate] = num+1
    
        subject = {'name':txtD[pair[0]][0],'bbox':[],'depth':[]}
        bbox = []
        bbox.extend([float(j) for j in txtD[pair[0]][1:5]])
        subject['bbox'] = bbox
        depth = []
        depth.extend([float(j) for j in txtD[pair[0]][5:8]])
        subject['depth'] = depth
        
        object = {'name':txtD[pair[1]][0],'bbox':[],'depth':[]}
        bbox = []
        bbox.extend([float(j) for j in txtD[pair[1]][1:5]])
        object['bbox'] = bbox
        depth = []
        depth.extend([float(j) for j in txtD[pair[1]][5:8]])
        object['depth'] = depth
    
        pairdic['object'] = object
        pairdic['subject'] = subject
    
        pairlist.append(pairdic)
        # print(img_idx)
    return pairlist, len(pairlistindx)

def main():
    txtpath = '/home/lx/DRNet/datasets/dataToAnno'
    num = 528
    ann_all = []
    anno_num = 0
    for img_idx in range(0,num):
          img = {'url': str(img_idx)+'.jpg', 'height': 480, 'width': 600, 'annotations':[]}
          annotations, anno_num_per = getanno(txtpath,img_idx)
          img['annotations'] = annotations
          ann_all.append(img)
          anno_num += anno_num_per
    # print(anno_num)
    # print(stat_num)
    json_data = json.dumps(ann_all, indent = 4)
    with open(txtpath + '/annotations.json', 'w') as f:
        f.write(json_data)

if __name__ == '__main__':
    main()
