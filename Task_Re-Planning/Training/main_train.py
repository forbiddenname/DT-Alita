import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import progressbar
from tensorboardX import SummaryWriter
from options import parse_args
from data_loader_order import create_dataloader
from models.GCN_LSTM import GCN_LSTM
from models.MLP_LSTM import MLP_LSTM
from models.MLP import MLP
from models.GCN_MLP import GCN_MLP
import pickle
import os
import sys
import time
from datetime import datetime as dt
import shutil
import numpy as np
from torchsummary import summary
from torch.nn.modules.module import _addindent
import torch
import numpy as np
# from util import accuracies, num_true_positives

writer = SummaryWriter('./trainTensorBoard319')


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

def train(model, criterion, optimizer, loader, epoch, args):
    model.train()
    loss = 0.
    acc = 0.
    acc1 = 0.
    acc_task = [0,0,0,0,0,0]
    task_count = [0,0,0,0,0,0]

    acc_action = np.ones((11,11))
    acc_object = np.ones((11,11))


    num_samples = 0
    num_samples1 = 0
    seq_len = args.seq_len
    mode = torch.tensor(1).cuda()
    act_obj_length = len(args.actions_categories)
    # batch_size = args.batchsize
    #bar = progressbar.ProgressBar(max_value=len(loader))
    for idx, data_batch in enumerate(loader):
        lengths_batch_all = torch.sum(torch.sum(data_batch[2], dim = 2),dim = 1)//2
        
        if args.model == 'gcnlstm' or args.model == 'gcnmlp':
            nodex = data_batch[0].x.cuda()
            # print(nodex.shape)
            edge_index = data_batch[0].edge_index.cuda()
            # print(edge_index.shape)
            edge_attr = data_batch[0].edge_attr.cuda()
            # print(edge_attr.shape)
            task_fea = data_batch[1].cuda()
            # print(task_fea.shape)
            task_sequence = data_batch[2].cuda()
            # print(task_sequence.shape)
            action_seq = task_sequence[:,0:seq_len,:]
            # print(action_seq.shape)
            lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
            # print(lengths_batch.shape)
            # summary(model,[(300,307),(2,489),(489,11),(30,1800),(30,15,22)],batch_size=-1)
            # summary(model,nodex, edge_index, edge_attr, task_fea, action_seq, lengths_batch, mode)

            actions, objects = model(nodex, edge_index, edge_attr, task_fea, action_seq, lengths_batch, mode)  ###输入前n-1个动作（不包括stop，stop）
        elif args.model == 'mlplstm':
            nodex = data_batch[0].cuda()
            task_fea = data_batch[1].cuda()
            task_sequence = data_batch[2].cuda()
            action_seq = task_sequence[:,0:seq_len,:]
            lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
            actions, objects = model(nodex, task_fea, action_seq, lengths_batch, mode)  ###输入前n-1个动作（不包括stop，stop）
        elif args.model == 'mlp':
            nodex = data_batch[0].cuda()
            task_fea = data_batch[1].cuda()
            task_sequence = data_batch[2].cuda()
            action_seq = task_sequence[:,0:seq_len,:]
            lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
            actions, objects = model(nodex, task_fea, action_seq, lengths_batch, mode)  ###输入前n-1个动作（不包括stop，stop）
        ##############计算当前batch中非0 的sequence
        deleterow = []
        for idx, length in enumerate(lengths_batch_all):
            if length < seq_len:
                for i in range(1,int(seq_len-length+2)):
                    deleterow.append(seq_len*(idx+1)-i)
    #             deleterow.append([seq_len*(idx+1)-i for i in range(1,int(seq_len-length+1))])
        index_batch = np.arange(0, seq_len*len(action_seq), 1, dtype=int)
        indices_batch = torch.tensor(np.delete(index_batch,deleterow)).cuda()
        # print(indices_batch)
        ##################################################################prediction
        # print(actions.shape)
        
        if args.model == 'mlp' or args.model == 'gcnmlp':
            actions_pre = actions
            objects_pre = objects
        else:
            diff = seq_len - actions.size(1)
            if diff > 0:
                action_size = actions.size(0)
                diff_data = torch.Tensor(np.zeros([action_size,diff,act_obj_length])).cuda()
                # print(diff_data.shape)
                actions_pre = torch.cat((actions,diff_data),1)
                objects_pre = torch.cat((objects,diff_data),1)
                actions_pre = actions_pre.reshape([-1,act_obj_length])
                objects_pre = objects_pre.reshape([-1,act_obj_length])
            else:
                actions_pre = actions.reshape([-1,act_obj_length])
                objects_pre = objects.reshape([-1,act_obj_length])

        # print(actions_pre.shape)
        actions_pre = torch.index_select(actions_pre,0, indices_batch)
        objects_pre = torch.index_select(objects_pre,0, indices_batch)
        
        # print(indices_batch)
        #### action_pre和object_pre行拼接 (batch_size * seq_len * 2)
        sequence_pre = torch.cat([actions_pre,objects_pre],0)
        #### action_pre和object_pre列拼接 (batch_size * seq_len，7维one_hot， 2)
        sequence_pre_1 = torch.cat([actions_pre,objects_pre],1).reshape([len(actions_pre),2,-1])
        sequence_pre_1 = sequence_pre_1.transpose(1,2)

        ##################################################################groundtruth
        
        action_seq_gt = task_sequence[:,1:seq_len+1,0:act_obj_length].argmax(dim=2).reshape([-1,1])
        action_seq_gt = torch.index_select(action_seq_gt,0, indices_batch)
        object_seq_gt = task_sequence[:,1:seq_len+1,act_obj_length:2*act_obj_length].argmax(dim=2).reshape([-1,1])
        object_seq_gt = torch.index_select(object_seq_gt,0, indices_batch)
        #### action和object行拼接 (batch_size * seq_len * 2)
        sequence_gt = torch.cat([action_seq_gt,object_seq_gt],0).squeeze()
        #### action和object列拼接 (batch_size * seq_len, 2)
        sequence_gt_1 = torch.cat([action_seq_gt,object_seq_gt],1)
        # print(sequence_pre_1.shape)
        # print(sequence_gt_1.shape)
        ##################################################################loss
        
        loss_batch_var = criterion(sequence_pre_1, sequence_gt_1)
        # print(loss_batch_var)
        # j = 0
        # for i,d in enumerate(lengths_batch_all):
        #     if i==0:
        #         loss_batch_var = criterion(sequence_pre_1[j:j + int(d)-1], sequence_gt_1[j:j + int(d)-1])
                
        #         j = j + int(d)-1
        #     else:
        #         loss_batch_var += criterion(sequence_pre_1[j:j + int(d)-1], sequence_gt_1[j:j + int(d)-1])
        #         print(sequence_pre_1[j:j + int(d)-1].shape)
        #         print(sequence_gt_1[j:j + int(d)-1].shape)
        #         j = j + int(d)-1


            #  if torch.eq(sequence_pre_2[range(j,j + int(d)-1)], sequence_gt_1[range(j,j + int(d)-1)]).sum() == 2*(d-1):
                #  acc1 = acc1+1
                #  j = j + int(d)-1
        loss_batch = loss_batch_var.item()
        num_samples += len(sequence_pre_1) ## 总样本数
        num_samples1 += len(lengths_batch_all)  ##总图数
        # loss += (len(action_seq_gt) * loss_batch)
        loss += (len(sequence_pre_1) * loss_batch)
        # print(len(lengths_batch_all))
        # print(sequence_pre_1.shape)
        # print(sequence_gt_1.shape)
        sequence_pre_2 = sequence_pre_1.argmax(dim=1)
        task_index = data_batch[3]
        # print(sequence_pre_2.shape)
        # print(sequence_pre_2)
        # print(lengths_batch)
        # print(lengths_batch_all)
        # print(task_index)

        for idx, data in enumerate(sequence_pre_2):
            # print(torch.eq(data, sequence_gt_1[idx]))
            if torch.eq(data, sequence_gt_1[idx]).sum() == 2:
                acc = acc + 1
        j = 0 
        for i,d in enumerate(lengths_batch_all):
            for k, data in enumerate(sequence_pre_2[j:j+int(d)-1]):
                eq = torch.eq(data,sequence_gt_1[j+k])
                # print(eq.shape)
                if eq.sum() == 2:
                    acc_task[int(task_index[i])] += 1 

                if eq[0] == 1:
                    acc_action[int(data[0]),int(data[0])] += 1
                else:
                    acc_action[int(sequence_gt_1[j+k][0]),int(data[0])] += 1
                if eq[1] == 1:
                    acc_object[int(data[1]),int(data[1])] += 1
                else:
                    acc_object[int(sequence_gt_1[j+k][1]),int(data[1])] += 1

            j = j+int(d)-1
            task_count[int(task_index[i])] += int(d)-1

            
            # if acc==len(sequence_gt_1):
            #     acc1 = acc1+1
        # j = 0
        # for i,d in enumerate(lengths_batch_all):
        #      print(torch.eq(sequence_pre_2[range(j,j + int(d)-1)], sequence_gt_1[range(j,j + int(d)-1)]))
        #      print(torch.eq(sequence_pre_2[range(j,j + int(d)-1)], sequence_gt_1[range(j,j + int(d)-1)]).sum())
        #      if torch.eq(sequence_pre_2[range(j,j + int(d)-1)], sequence_gt_1[range(j,j + int(d)-1)]).sum() == 2*(d-1):
        #          acc1 = acc1+1
        #          j = j + int(d)-1
        # acc += torch.eq(sequence_pre_1.argmax(dim=1), sequence_gt_1).sum().float().item()
        
        optimizer.zero_grad()
        loss_batch_var.backward()
        optimizer.step()
 
        #bar.update(idx)
    # print(num_samples)
    # print(num_samples1)
    loss /= num_samples
    acc /= num_samples
    acc1 /= num_samples1
    print(num_samples)
    for i in range(0,len(acc_task)):
        acc_task[i] /= task_count[i]
        acc_task[i] = np.round(acc_task[i],3)
    # acc_object/acc_object.sum(axis=1)[:, np.newaxis]

    return loss, acc, acc1,acc_task, acc_action/acc_action.sum(axis=1)[:, np.newaxis], acc_object/acc_object.sum(axis=1)[:, np.newaxis]


def test(model, criterion, loader, epoch, args):
    model.eval()
    loss = 0.
    acc = 0.
    num_samples = 0
    seq_len = args.seq_len
    mode = torch.tensor(1).cuda()
    act_obj_length = len(args.actions_categories)
    act_length = len(args.actions_categories)
    obj_length = len(args.objects_categories)
    # batch_size = args.batchsize

    with torch.no_grad():  #不计算梯度,不反向传播
        # bar = progressbar.ProgressBar(max_value=len(loader))
        for idx, data_batch in enumerate(loader):
            lengths_batch_all = torch.sum(torch.sum(data_batch[2], dim = 2),dim = 1)//2
            if args.model == 'gcnlstm' or args.model == 'gcnmlp':
                nodex = data_batch[0].x.cuda()
                edge_index = data_batch[0].edge_index.cuda()
                edge_attr = data_batch[0].edge_attr.cuda()
                task_fea = data_batch[1].cuda()
                task_sequence = data_batch[2].cuda()
                action_seq = task_sequence[:,0:seq_len,:]
                lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
                actions, objects = model(nodex, edge_index, edge_attr, task_fea, action_seq, lengths_batch, mode)  ###输入前n-1个动作（不包括stop，stop）
            elif args.model == 'mlplstm':
                nodex = data_batch[0].cuda()
                task_fea = data_batch[1].cuda()
                task_sequence = data_batch[2].cuda()
                action_seq = task_sequence[:,0:seq_len,:]
                lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
                actions, objects = model(nodex, task_fea, action_seq, lengths_batch, mode)  ###输入前n-1个动
            elif args.model == 'mlp':
                nodex = data_batch[0].cuda()
                task_fea = data_batch[1].cuda()
                task_sequence = data_batch[2].cuda()
                action_seq = task_sequence[:,0:seq_len,:]
                lengths_batch = (torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2)
                actions, objects = model(nodex, task_fea, action_seq, lengths_batch, mode) 
                ##############计算当前batch中非0 的sequence
            deleterow = []
            for idx, length in enumerate(lengths_batch_all):
                if length < seq_len:
                    for i in range(1,int(seq_len-length+2)):
                        deleterow.append(seq_len*(idx+1)-i)
        #             deleterow.append([seq_len*(idx+1)-i for i in range(1,int(seq_len-length+1))])
            index_batch = np.arange(0, seq_len*len(action_seq), 1, dtype=int)
            indices_batch = torch.tensor(np.delete(index_batch,deleterow)).cuda()
            # print(indices_batch)
            ##################################################################prediction
            # print(actions.shape)
            if args.model == 'mlp' or args.model == 'gcnmlp':
                actions_pre = actions
                objects_pre = objects
            else:
                diff = seq_len - actions.size(1)
                if diff > 0:
                    action_size = actions.size(0)
                    diff_data = torch.Tensor(np.zeros([action_size,diff,act_obj_length])).cuda()
                    # print(diff_data.shape)
                    actions_pre = torch.cat((actions,diff_data),1)
                    objects_pre = torch.cat((objects,diff_data),1)
                    actions_pre = actions_pre.reshape([-1,act_obj_length])
                    objects_pre = objects_pre.reshape([-1,act_obj_length])
                else:
                    actions_pre = actions.reshape([-1,act_obj_length])
                    objects_pre = objects.reshape([-1,act_obj_length])

            # print(actions_pre.shape)
            actions_pre = torch.index_select(actions_pre,0, indices_batch)
            objects_pre = torch.index_select(objects_pre,0, indices_batch)
            
            # print(indices_batch)
            #### action_pre和object_pre行拼接 (batch_size * seq_len * 2)
            sequence_pre = torch.cat([actions_pre,objects_pre],0)
            #### action_pre和object_pre列拼接 (batch_size * seq_len，7维one_hot， 2)
            sequence_pre_1 = torch.cat([actions_pre,objects_pre],1).reshape([len(actions_pre),2,-1])
            sequence_pre_1 = sequence_pre_1.transpose(1,2)

            ##################################################################groundtruth
            # action_seq_gt = data_batch[2][:,1:10,0:7].argmax(dim=2)
            # object_seq_gt = data_batch[2][:,1:10,7:14].argmax(dim=2)
        
            # print(data_batch[2])
            # print(task_sequence)
            # action_seq_gt = data_batch[2]
            action_seq_gt = task_sequence[:,1:seq_len+1,0:act_length].argmax(dim=2).reshape([-1,1])
            # print(action_seq_gt.shape)
            # print(indices_batch)
            action_seq_gt = torch.index_select(action_seq_gt,0, indices_batch)
            object_seq_gt = task_sequence[:,1:seq_len+1,act_length:act_length + obj_length].argmax(dim=2).reshape([-1,1])
            object_seq_gt = torch.index_select(object_seq_gt,0, indices_batch)
            #### action和object行拼接 (batch_size * seq_len * 2)
            sequence_gt = torch.cat([action_seq_gt,object_seq_gt],0).squeeze()
            #### action和object列拼接 (batch_size * seq_len, 2)
            sequence_gt_1 = torch.cat([action_seq_gt,object_seq_gt],1)

            ##################################################################loss
            # print(sequence_pre_1.shape)
            # print(sequence_gt_1.shape)
            loss_batch_var = criterion(sequence_pre_1, sequence_gt_1)
            loss_batch = loss_batch_var.item()
            num_samples += len(sequence_pre_1) ## 总样本数
            # loss += (len(action_seq_gt) * loss_batch)
            loss += (len(sequence_pre_1) * loss_batch + 1e-6)

            for idx, data in enumerate(sequence_pre_1.argmax(dim=1)):
                if torch.eq(data, sequence_gt_1[idx]).sum() == 2:
                    acc = acc + 1
            # acc += torch.eq(sequence_pre_1.argmax(dim=1), sequence_gt_1).sum().float().item()
            
            # optimizer.zero_grad()
            # loss_batch_var.backward()
            # optimizer.step()
    
            #bar.update(idx)

        loss /= num_samples
        acc /= num_samples
    
        return loss, acc


def main():

    args = parse_args()

    # root = '/home/lx/TaskPlaning/datasets'
    # log_dir = '/home/lx/TaskPlaning/'
    # batch_size = 5
    # n_epochs = 30
    # learning_rate = 0.001
    dataloader_train, dataloader_valid, dataloader_test = create_dataloader(args)
    
    
    print('%d batches of training examples' % len(dataloader_train))
    print('%d batches of validation examples' % len(dataloader_valid))
    print('%d batches of testing examples' % len(dataloader_test))
    print ('learning_rate:', args.learning_rate)
    print ('batchsize:', args.batchsize)

    
    # if args.model == 'drnet':
    #     model =DRNet_depth(phrase_encoder, args.feature_dim, args.num_layers)
    # elif args.model == 'vtranse':
    #     model = VtransE(phrase_encoder, args.visual_feature_size, args.predicate_embedding_dim)
    # elif args.model == 'vipcnn':
    #     model = VipCNN(roi_size=args.roi_size, backbone=args.backbone)
    # else:
    #     model = PPRFCN(backbone=args.backbone)
    # gcnlstm = GCN_LSTM().cuda()

    if args.model == 'gcnlstm':
        model = GCN_LSTM(args).cuda()
    elif args.model == 'mlplstm':
        model = MLP_LSTM(args).cuda()
    elif args.model == 'mlp':
        model = MLP(args).cuda()
    elif args.model == 'gcnmlp':
        model = GCN_MLP(args).cuda()
    
    print(model)
    print(torch_summarize(model))
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.l2)

    scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
    # if args.train_split == 'train':
    #     scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
    # else:
    #     scheduler = StepLR(optimizer, step_size=args.patience, gamma=0.1) 

    start_epoch = 0

    if args.resume != None:
        print(' => loading model checkpoint from %s..' % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    

    best_acc = -1.

    # time2 = dt.strftime(dt.now(),'%Y%m%d%H%M%S')
    # filetxt = '/home/lx/TaskPlaning/train_results/'+args.model+args.stiff+args.rel+args.label+'/'+time2+'.txt'
    # os.makedirs(os.path.dirname(filetxt), exist_ok=True)

    for epoch in range(start_epoch, start_epoch + args.n_epochs):

        print('epoch #%d' % epoch)
   
        print('training..')

        loss, acc, acc1, acc_task, confusionM_action, confusionM_object  = train(model, criterion, optimizer, dataloader_train, epoch, args)

        
        # fh2 = open(filetxt, 'a', encoding='utf-8')
        # fh2.write(str(epoch)+'\r\n')
        # fh2.write(str(acc)+'\r\n')
        # fh2.write(str(acc_task)+'\r\n')
        # fh2.write(str(confusionM_action)+'\r\n')
        # fh2.write(str(confusionM_object)+'\r\n')
        

        print('\n\ttraining loss = %.4f' % loss)
        print('\ttraining accuracy = %.3f' % acc)
        # print(acc_task)
        # print(confusionM_action)
        # print(confusionM_object)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        # writer.add_scalar('train/acc', acc1, epoch)
        # if args.train_split != 'train_valid':
        if epoch % 5 == 0:
            print('validating..')
            torch.cuda.synchronize()
            start = time.time()
            loss, acc = test(model, criterion, dataloader_valid, epoch, args)
            torch.cuda.synchronize()
            end = time.time()
            dtime = ((end - start)/len(dataloader_valid)/args.batchsize)
            print('\n\tvalidation loss = %.4f' % loss)
            print('\tvalidation accuracy = %.3f' % acc)
            print('\tvalidation time per input = %.3f' % dtime )
            writer.add_scalar('test/loss', loss, epoch)
            writer.add_scalar('test/acc', acc, epoch)
        #     # for predi in acc:
        #     #     if predi != 'overall':
        #     #         print('\t\t%s: %.3f' % (predi, acc[predi]))
        if args.cksave == 'true':
            checkpoint_filename = os.path.join(args.log_dir, 'checkpoints/model_%.3f_%02d.pth' % (acc,epoch))
            model.cpu()
            torch.save({'epoch': epoch + 1,
                        'args': args,
                        'state_dict': model.state_dict(),
                        'accuracy': acc,
                        'optimizer' : optimizer.state_dict(),
                    }, checkpoint_filename)
            model.cuda()
     
        # if args.train_split != 'train_valid' and best_acc < acc:
        #     best_acc = acc
        #     shutil.copyfile(checkpoint_filename, os.path.join(args.log_dir, 'checkpoints/model_best.pth'))
        #     shutil.copyfile(os.path.join(args.log_dir, 'predictions/pred_%02d.pickle' % epoch), 
        #                     os.path.join(args.log_dir, 'predictions/pred_best.pickle'))

        # if args.train_split == 'train':
        #     scheduler.step(loss)
        # else:
        #     scheduler.step()
        scheduler.step(loss)
    writer.add_graph(model, model)
    print('testing..')
    loss, acc = test(model, criterion, dataloader_test, None, args)
    print('\n\ttesting loss = %.4f' % loss)
    print('\ttesting accuracy = %.3f' % acc)
    writer.close()
    # fh2.close()
    
    # print('\ttesting accuracy = %.3f' % acc['overall'])
    # for predi in acc:
    #     if predi != 'overall':
    #         print('\t\t%s: %.3f' % (predi, acc[predi]))


if __name__ == '__main__':
    main()
