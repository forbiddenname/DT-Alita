# root = '/home/lx/TaskPlaning/datasets'
# phrase_encoder = RecurrentPhraseEncoder(100, 100)


########## 将节点按照顺序排，填满8个

import torch
import torch_geometric
import numpy as np
import os
# from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
# from torch.utils.data import Dataset, DataLoader
from util import phrase2vec, onehot

def Datasets_gcnlstm(args):
        root = args.datapath
        data_list = []
        raw_data_root = root + '/raw_data/'
        txt_list = os.listdir(raw_data_root)
        txt_list.sort(key=lambda x:int(x[:-4]))
        task_root = root + '/task_action/'

        task_count = [0,0,0,0,0,0]
        action_count = np.ones((11,11))
        object_count = np.ones((11,11))
    ################# graph features
        for txt_name1 in txt_list:
            txt_name = os.path.join(raw_data_root, txt_name1)
            with open(txt_name) as f:
        #         listobject = []
                node_attr = []
                node_spalenth = args.node_spalenth   ##8为包含刚度的特征维数
                node_attr_order = np.zeros((len(args.object_raw_categories),300+node_spalenth))    ## 8为当前固定的节点个数，需要修改； 307为 300word embedding + 7位置信息
                node_label = []
                edge_index = []
                edge_attr  = []
                index = 0 #当前文件下第几行                  
                for f1 in f.readlines():
                    f1 = f1.replace("\n", "").strip().split(' ')
                    node_attr_label_vec = phrase2vec(f1[0],1,300).flatten() ## 是否需要考虑物体间的种类关系，如果不考虑的话，可以换成one-hot？因为其实也没有多少类物体
                    node_attr_spatial_vec = np.array([float(i) for i in f1[1:1+node_spalenth]]) ##[1:8]共7维数为4二维+3三维空间特征维数，否则[1:9]共8维数为4二维+3三维空间特征维数+1刚度特征
                    node_attr_vec = np.hstack((node_attr_label_vec,node_attr_spatial_vec)) 
                    node_label.append(f1[0])                  
                    node_attr.append(node_attr_vec)
                    node_attr_order[args.object_raw_categories.index(f1[0])] = node_attr_vec                  
                    ## edge_index
                    for i, data in enumerate(f1[1+node_spalenth:len(f1)]):
                        if int(data) != 0:
                            edge_index.append([index,i])
                            edge_attr.append(onehot(int(data)-1,11))    # 注意由于nothing为0，因此 11表示空间关系数，即边的特征维数，而没有用len(predicate_categories）#                   edge_index_order.append([object_raw_categories.index(node_label(index)),object_raw_categories.index(node_label(i))])
                    index = index + 1               
                edge_index = np.transpose(edge_index)
                edge_index_order = np.array(edge_index, copy=True)  
                # print(txt_name)             
                for i in range(0,len(edge_index)):
                    for j in range(0,len(edge_index[0])):
                        edge_index_order[i,j] = args.object_raw_categories.index(node_label[edge_index[i,j]])

        ################# ground truth
            with open(task_root + txt_name1) as f:
        #         task_sequence_vec = []
        #         index1 = 0
        #         for f1 in f.readlines():
        #             f1 = f1.replace("\n", "").strip()
        #             if index1 == 0:
        #                 task_fea_vec = phrase2vec(str(f1),6,300).flatten()   # 6表示任务名称最大字符数，能否改成one-hot？
        # #                 print(f1)
        #             else:
        #                 action_name = f1.split(' ')[0]
        #                 action_index = actions_categories.index(action_name)
        #                 action_vec = onehot(action_index, len(actions_categories))

        #                 object_name = f1.split(' ')[1]
        #                 object_index = object_categories.index(object_name)
        #                 object_vec = onehot(object_index, len(object_categories))

        #                 task_sequence_vec.append(np.hstack((action_vec,object_vec)))

        #             index1 = index1 + 1
                task_sequence_vec = np.zeros([args.seq_len+1,args.actobjlen]) ##seq_len+1为样本中最多的actions_seq，包括stop stop
                index1 = 0
                # print(f)
                for f1 in f.readlines():
                    
                    f1 = f1.replace("\n", "").strip()
                    if index1 == 0:
                        task_index = args.task_categories.index(str(f1))
                        if args.task_encode == 'word2vec':
                            task_fea_vec = phrase2vec(str(f1),args.maxtasklen,300).flatten()   # 4表示任务名称最大字符数，能否改成one-hot？
                        elif args.task_encode == 'onehot':
                            task_fea_vec = onehot(task_index,len(args.task_categories))
                        
                        task_count[task_index] += 1
                        

        #                 print(f1)
                    else:
                        
                        action_name = f1.split(' ')[0]
                        # print(action_name)
                        # action_index = actions_categories.index(action_name)
                        action_index = int(action_name)
                        action_vec = onehot(action_index, len(args.actions_categories))
                        # print(action_vec)
                        action_count[task_index,action_index] += 1

                        object_name = f1.split(' ')[1]
                        # print(object_name )
                        # object_index = object_categories.index(object_name)
                        object_index = int(object_name)
                        object_vec = onehot(object_index, len(args.objects_categories))

                        object_count[task_index,object_index] += 1
                        # print(object_vec)
                        
                        task_sequence_vec[index1-1] = np.hstack((action_vec,object_vec))
                    index1 = index1 + 1
                # print(index1)

            data = Data(x = torch.tensor(node_attr_order,dtype=torch.float32),edge_index = torch.tensor(edge_index_order,dtype=torch.long), edge_attr = torch.tensor(edge_attr,dtype=torch.float32))
            data_list.append([data, torch.tensor(task_fea_vec,dtype=torch.float32), torch.tensor(task_sequence_vec,dtype=torch.float32), torch.tensor(task_index,dtype=torch.float32)])
        # print(task_count)
        # print(action_count)
        # print(object_count)
        return data_list
        
def Datasets_mlplstm(args):
        root = args.datapath
        data_list = []
        raw_data_root = root + '/raw_data/'
        txt_list = os.listdir(raw_data_root)
        txt_list.sort(key=lambda x:int(x[:-4]))
        task_root = root + '/task_action/'
        node_spalenth = args.node_spalenth   ##8为包含刚度的特征维数
        object_raw_len=len(args.object_raw_categories)
    ################# graph features
        for txt_name1 in txt_list:
            txt_name = os.path.join(raw_data_root, txt_name1)
            with open(txt_name) as f:
        #         listobject = []
                node_attr = []
                
                node_attr_order = np.zeros((object_raw_len,300+node_spalenth))    ## 8为当前固定的节点个数，需要修改； 307为 300word embedding + 7位置信息
                node_label = []
                edge_index = []
                edge_attr  = []
                edge_attr_order = np.zeros((object_raw_len,object_raw_len))
                index = 0 #当前文件下第几行                  
                for f1 in f.readlines():
                    f1 = f1.replace("\n", "").strip().split(' ')
                    node_attr_label_vec = phrase2vec(f1[0],1,300).flatten() ## 是否需要考虑物体间的种类关系，如果不考虑的话，可以换成one-hot？因为其实也没有多少类物体
                    node_attr_spatial_vec = np.array([float(i) for i in f1[1:1+node_spalenth]]) ##[1:8]共7维数为4二维+3三维空间特征维数，否则[1:9]共8维数为4二维+3三维空间特征维数+1刚度特征
                    node_attr_vec = np.hstack((node_attr_label_vec,node_attr_spatial_vec)) 
                    node_label.append(f1[0])                  
                    node_attr.append(node_attr_vec)
                    node_attr_order[args.object_raw_categories.index(f1[0])] = node_attr_vec                  
                    ## edge_index
                    for i, data in enumerate(f1[1+node_spalenth:len(f1)]):
                        if int(data) != 0:
                            edge_index.append([index,i])
                            edge_attr.append(int(data))
                            # edge_attr.append(onehot(int(data)-1,11))    # 注意由于nothing为0，因此 11表示空间关系数，即边的特征维数，而没有用len(predicate_categories）#                   edge_index_order.append([object_raw_categories.index(node_label(index)),object_raw_categories.index(node_label(i))])
                    index = index + 1               
                edge_index = np.transpose(edge_index)
                edge_index_order = np.array(edge_index, copy=True)               
                for i in range(0,len(edge_index)):
                    for j in range(0,len(edge_index[0])):
                        edge_index_order[i,j] = args.object_raw_categories.index(node_label[edge_index[i,j]])
                
                for i in range(0,edge_index_order.shape[1]):
#                     print(edge_attr[i])
                    edge_attr_order[edge_index_order[:,i][0],edge_index_order[:,i][1]] = edge_attr[i]
                node_edge_attr = np.hstack((node_attr_order,edge_attr_order))

        ################# ground truth
            with open(task_root + txt_name1) as f:
                task_sequence_vec = np.zeros([args.seq_len+1,args.actobjlen]) ##seq_len+1为样本中最多的actions_seq，包括stop stop
                index1 = 0
                # print(f)
                for f1 in f.readlines():
                    
                    f1 = f1.replace("\n", "").strip()
                    if index1 == 0:
                        task_fea_vec = phrase2vec(str(f1),args.maxtasklen,300).flatten()   # 4表示任务名称最大字符数，能否改成one-hot？
                        task_index = args.task_categories.index(str(f1))
        #                 print(f1)
                    else:
                        
                        action_name = f1.split(' ')[0]
                        action_index = int(action_name)
                        action_vec = onehot(action_index, len(args.actions_categories))
                        object_name = f1.split(' ')[1]
                        object_index = int(object_name)
                        object_vec = onehot(object_index, len(args.objects_categories))                        
                        task_sequence_vec[index1-1] = np.hstack((action_vec,object_vec))
                    index1 = index1 + 1
            data_list.append([torch.tensor(node_edge_attr,dtype=torch.float32), torch.tensor(task_fea_vec,dtype=torch.float32), torch.tensor(task_sequence_vec,dtype=torch.float32),torch.tensor(task_index,dtype=torch.float32)])

        return data_list

def create_dataloader(args):
    
    batchsize = args.batchsize
    trainratio = args.trainratio
    if args.model == 'gcnlstm' or args.model == 'gcnmlp':
        datasets = Datasets_gcnlstm(args)
    elif args.model == 'mlplstm' or args.model == 'mlp' or args.model == 'nn':
        datasets = Datasets_mlplstm(args)

    if args.model == 'nn':
        return datasets
        # return DataLoader(datasets,len(datasets), shuffle = True)
    else:
        train_size = int(trainratio * len(datasets))
        test_size = (len(datasets) - train_size)//2
        validation_size = len(datasets) - train_size - test_size
        
        train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(datasets, [train_size, test_size, validation_size])
        
        train_loader = DataLoader(train_dataset,batchsize, shuffle = True)
        test_loader = DataLoader(test_dataset,batchsize, shuffle = True)
        valid_loader = DataLoader(valid_dataset,batchsize, shuffle = True)
        
        return train_loader, test_loader, valid_loader