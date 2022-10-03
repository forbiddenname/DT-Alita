# root = '/home/lx/TaskPlaning/datasets'
# phrase_encoder = RecurrentPhraseEncoder(100, 100)
import torch
import torch_geometric
import numpy as np
import os
# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
# from torch.utils.data import Dataset, DataLoader
from util import phrase2vec, onehot

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
                    'hold'] ###touch or hold

actions_categories = ['grasp',
                  'place',
                  'pour',
                  'push',
                  'start',
                  'stop']

object_categories = ['sponge',
                 'cup',
                 'apple',
                 'banana',
                 'orange',
                 'start',
                 'stop']
def TaskPlaningDatasets(root):
        data_list = []
        raw_data_root = root + '/raw_data/'
        txt_list = os.listdir(raw_data_root)
        txt_list.sort(key=lambda x:int(x[:-4]))
        task_root = root + '/task_action/'

    ################# graph features
        for txt_name1 in txt_list:
            txt_name = os.path.join(raw_data_root, txt_name1)
            with open(txt_name) as f:
        #         listobject = []
                node_attr = []
                edge_index = []
                edge_attr = []
                index = 0 #当前文件下第几行
                for f1 in f.readlines():
                    f1 = f1.replace("\n", "").strip().split(' ')
                    node_attr_label_vec = phrase2vec(f1[0],1,300).flatten() ## 是否需要考虑物体间的种类关系，如果不考虑的话，可以换成one-hot？因为其实也没有多少类物体
                    node_attr_spatial_vec = np.array([float(i) for i in f1[1:8]])
                    node_attr_vec = np.hstack((node_attr_label_vec,node_attr_spatial_vec)) ##307维？需要处理
                    node_attr.append(node_attr_vec)
                    ## edge_index
                    for i, data in enumerate(f1[8:len(f1)]):
                        if int(data) != 0:
                            edge_index.append([float(index),float(i)])
                            edge_attr.append(onehot(int(data)-1,11))    # 注意由于nothing为0，因此 11表示空间关系数，即边的特征维数
                    index = index + 1
                edge_index = np.transpose(edge_index)

        ################# ground truth
            with open(task_root + txt_name1) as f:
                task_sequence_vec = []
                index1 = 0
                for f1 in f.readlines():
                    f1 = f1.replace("\n", "").strip()
                    if index1 == 0:
                        task_fea_vec = phrase2vec(str(f1),6,300)   # 6表示任务名称最大字符数，能否改成one-hot？
        #                 print(f1)
                    else:
                        action_name = f1.split(' ')[0]
                        action_index = actions_categories.index(action_name)
                        action_vec = onehot(action_index, len(actions_categories))

                        object_name = f1.split(' ')[1]
                        object_index = object_categories.index(object_name)
                        object_vec = onehot(object_index, len(object_categories))

                        task_sequence_vec.append(np.hstack((action_vec,object_vec)))

                    index1 = index1 + 1

            data = Data(x = torch.tensor(node_attr,dtype=torch.float32),edge_index = torch.tensor(edge_index,dtype=torch.long), edge_attr = torch.tensor(edge_attr,dtype=torch.float32))
            data_list.append([data, torch.tensor(task_fea_vec,dtype=torch.float32), torch.tensor(task_sequence_vec,dtype=torch.float32)])

        return data_list
    
# def __len__(self):
#     return len(data_list)
# #         data_loader = DataLoader(data_list,batch_size = 5, shuffle = True)

        
    
def create_dataloader(root, batchsize):
    datasets = TaskPlaningDatasets(root)
    train_size = int(0.8 * len(datasets))
    test_size = (len(datasets) - train_size)//2
    validation_size = len(datasets) - train_size - test_size
    
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(datasets, [train_size, test_size, validation_size])
    
    train_loader = DataLoader(train_dataset,batchsize, shuffle = True)
    test_loader = DataLoader(test_dataset,batchsize, shuffle = True)
    valid_loader = DataLoader(valid_dataset,batchsize, shuffle = True)
    
    return train_loader, test_loader, valid_loader