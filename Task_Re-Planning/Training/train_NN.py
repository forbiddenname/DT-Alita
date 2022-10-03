import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime as dt
from options import parse_args
from torch_geometric.data import DataLoader

from data_loader_order import create_dataloader



def main():
    args = parse_args()
    datasets = create_dataloader(args)
    act_obj_length = len(args.actions_categories)
    start_epoch = 0

    time2 = dt.strftime(dt.now(),'%Y%m%d%H%M%S')
    filetxt = '/home/lx/TaskPlaning/train_results/'+args.model+args.stiff+args.rel+args.label+'/'+time2+'.txt'
    os.makedirs(os.path.dirname(filetxt), exist_ok=True)
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

        acc_task = [0,0,0,0,0,0]
        task_count = [0,0,0,0,0,0]

        acc_action = np.ones((11,11))
        acc_object = np.ones((11,11))
        acc = 0.
        num = 0.
        
        for idx, data_batch in enumerate(test_loader):
            task_index = data_batch[3]
            featurex = torch.cat([data_batch[0].reshape([1,-1]),data_batch[1]],1)
            dis = 1000
            for idy, data_batch_train in enumerate(train_loader):
                featurey = torch.cat([data_batch_train[0].reshape([1,-1]),data_batch_train[1]],1)
            
                distance = F.pairwise_distance(featurex, featurey, p=2)
                if distance < dis:
                    dis = distance
                    atcion_obj = data_batch_train[2].squeeze(0).reshape([-1,2,act_obj_length])
        #     print(data_batch[2].shape)
            atcion_obj_t = data_batch[2].squeeze(0).reshape([-1,2,act_obj_length])
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
            # eq = torch.eq(atcion_obj_tru.argmax(dim=2), atcion_obj_pre.argmax(dim=2))
            atcion_obj_tru1=atcion_obj_tru.argmax(dim=2)
            atcion_obj_pre1=atcion_obj_pre.argmax(dim=2)

            for idx, data in enumerate(atcion_obj_pre1):
            # print(torch.eq(data, sequence_gt_1[idx]))
                eq = torch.eq(data, atcion_obj_tru1[idx])
                if eq.sum() == 2:
                    acc = acc + 1
                    acc_task[int(task_index)] += 1 

                if eq[0] == 1:
                    acc_action[int(data[0]),int(data[0])] += 1
                else:
                    acc_action[int(atcion_obj_tru1[idx][0]),int(data[0])] += 1
                if eq[1] == 1:
                    acc_object[int(data[1]),int(data[1])] += 1
                else:
                    acc_object[int(atcion_obj_tru1[idx][1]),int(data[1])] += 1            
                task_count[int(task_index)] += 1
                
            num += atcion_obj_t_sum

            # b = torch.sum(eq,dim = 1)
            # for data in b:
            #     if data == 2:
            #         acc= acc+1  
            # num += atcion_obj_t_sum

         


        for i in range(0,len(acc_task)):
            acc_task[i] /= task_count[i]
            acc_task[i] = np.round(acc_task[i],3)
        acc/= num
        confusionM_action = acc_action/acc_action.sum(axis=1)[:, np.newaxis]
        confusionM_object = acc_object/acc_object.sum(axis=1)[:, np.newaxis]
        print(acc,acc_task,confusionM_action, confusionM_object)
        fh2 = open(filetxt, 'a', encoding='utf-8')
        fh2.write(str(epoch)+'\r\n')
        fh2.write(str(acc)+'\r\n')
        fh2.write(str(acc_task)+'\r\n')
        fh2.write(str(confusionM_action)+'\r\n')
        fh2.write(str(confusionM_object)+'\r\n')
if __name__ == '__main__':
    main()







        
