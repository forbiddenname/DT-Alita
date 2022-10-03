import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
# from .GN_Layers import Edge_node_Model
from .recurrent_phrase_encoder import RecurrentPhraseEncoder
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
from data_loader_order import create_dataloader
import torch.nn.functional as F

def main():
    args = parse_args()
    datasets = create_dataloader(args)
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        print('epoch #%d' % epoch)
        torch.cuda.synchronize()
        start = time.time()
        loss = 0.
        acc = 0.
        train_size = int(0.8 * len(datasets))
        test_size = (len(datasets) - train_size)
        train_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset,1, shuffle = True)
        test_loader = DataLoader(test_dataset,1, shuffle = True)

        acc = 0.
        num = 0.
        for idx, data_batch in enumerate(test_loader):
            featurex = torch.cat([data_batch[0].reshape([1,-1]),data_batch[1]],1)
            dis = 1000
            for idy, data_batch_train in enumerate(train_loader):
                featurey = torch.cat([data_batch_train[0].reshape([1,-1]),data_batch_train[1]],1)
            
                distance = F.pairwise_distance(featurex, featurey, p=2)
                if distance < dis:
                    dis = distance
                    atcion_obj = data_batch_train[2].squeeze(0).reshape([-1,2,9])
        #     print(data_batch[2].shape)
            atcion_obj_t = data_batch[2].squeeze(0).reshape([-1,2,9])
        #     print(atcion_obj_t.shape)
            atcion_obj_t_sum = torch.sum(atcion_obj_t)/2
            atcion_obj_p_sum = torch.sum(atcion_obj)/2
        #     if atcion_obj_t_sum > atcion_obj_p_sum:
        #         atcion_obj_pre = atcion_obj[0:atcion_obj_t_sum,:,:]
        #     else:
        #     atcion_obj_t_sum = torch.sum(atcion_obj_t_sum,dim = 1)/2
            atcion_obj_pre = atcion_obj[0:int(atcion_obj_t_sum),:,:]
            atcion_obj_tru = atcion_obj_t[0:int(atcion_obj_t_sum),:,:]
        #     print(atcion_obj_t_sum)
            a = torch.eq(atcion_obj_tru.argmax(dim=2), atcion_obj_pre.argmax(dim=2))
            b = torch.sum(a,dim = 1)
            for data in b:
                if data == 2:
                    acc= acc+1  
            num += atcion_obj_t_sum
        acc/= num
        print(acc)








        
