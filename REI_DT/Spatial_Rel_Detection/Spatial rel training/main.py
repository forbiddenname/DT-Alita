import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import progressbar
from options import parse_args
from dataloader import create_dataloader
from dataloader_depth import create_dataloader_depth
from models.recurrent_phrase_encoder import RecurrentPhraseEncoder
import pickle
import os
import sys
import time
from datetime import datetime

from models.drnet import DRNet
from models.drnet_depth1 import DRNet_depth
import shutil



def train(model, criterion, optimizer, loader, epoch, args):

    model.train()
    loss = 0.
    acc = 0.
    num_samples = 0
    print(len(loader))

    for idx, data_batch in enumerate(loader):
        predicate = data_batch['predicate'].cuda()
        subj_batch_var = data_batch['subject']['embedding'].cuda()
        obj_batch_var = data_batch['object']['embedding'].cuda()
        img = data_batch['bbox_img'].cuda()
        mask_batch_var = data_batch['bbox_mask'].cuda()
        multispa_batch = data_batch['multispa'].cuda()

        if args.depth_on == 'on':
            subj_batch_depth = data_batch['subject']['depth'].cuda()
            obj_batch_depth = data_batch['object']['depth'].cuda()
            output = model(subj_batch_var, obj_batch_var, img, mask_batch_var, multispa_batch,subj_batch_depth, obj_batch_depth, args)
        else:
            output = model(subj_batch_var, obj_batch_var, img, mask_batch_var, multispa_batch,args)


        num_samples += len(data_batch['predicate'])
        gt = predicate.argmax(dim=1)
        
        loss_batch_var = criterion(output, gt)
        loss_batch = loss_batch_var.item()
        loss += (len(data_batch['predicate']) * loss_batch)
        
        pred = output.argmax(dim=1)
        # gt = predicate.argmax(dim=1)
        acc += torch.eq(pred, gt).sum().float().item()
        
        for i in range(0,gt.size(0)):
            rellist_count[gt[i].item()]+=1

        for i in range(0,pred.size(0)):
            if torch.eq(pred, gt)[i].item():
                acc_list[pred[i].item()]+=1
        
        
        optimizer.zero_grad()
        loss_batch_var.backward()
        optimizer.step()
 
        #bar.update(idx)
    
    loss /= num_samples
    acc /= (num_samples / 100.)

    return loss, acc


def test(split, model, criterion, loader, epoch, args):
    model.eval()
    loss = 0.
    _ids = []
    predictions = []
    acc = 0.
    num_samples = 0

    with torch.no_grad():  #不计算梯度,不反向传播
        # bar = progressbar.ProgressBar(max_value=len(loader))
        for idx, data_batch in enumerate(loader):
            subj_batch_var = data_batch['subject']['embedding'].cuda()
            obj_batch_var = data_batch['object']['embedding'].cuda()
            predicate = data_batch['predicate'].cuda()
            img = data_batch['bbox_img'].cuda()
            mask_batch_var = data_batch['bbox_mask'].cuda()
            multispa_batch = data_batch['multispa'].cuda()
 
            if args.depth_on == 'on':
                subj_batch_depth = data_batch['subject']['depth'].cuda()
                obj_batch_depth = data_batch['object']['depth'].cuda()
                output = model(subj_batch_var, obj_batch_var, img, mask_batch_var, multispa_batch,subj_batch_depth, obj_batch_depth, args)
            else:
                output = model(subj_batch_var, obj_batch_var, img, mask_batch_var, multispa_batch,args)

            predictions.append(output)
     
            # if 'label' in data_batch: 
            #     label_batch_var = torch.squeeze(data_batch['label']).cuda()
            #     loss_batch_var = criterion(output, label_batch_var)
            #     loss_batch = loss_batch_var.item()
            #     loss += (len(data_batch['label']) * loss_batch)

            num_samples += len(data_batch['predicate'])
            gt = predicate.argmax(dim=1)
        
            loss_batch_var = criterion(output, gt)
            loss_batch = loss_batch_var.item()
            loss += (len(data_batch['predicate']) * loss_batch)
        
            pred = output.argmax(dim=1)
            # gt = predicate.argmax(dim=1)
            acc += torch.eq(pred, gt).sum().float().item()

        loss /= num_samples
        acc /= (num_samples / 100.)
        return loss, acc


def main():
    time1 = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    fh = open('/home/lx/DRNet/experiment/spatialrel/rel'+ time1 +'.txt', 'w', encoding='utf-8')
    args = parse_args()
    if args.custom_on == 'on':
        dataloader_train = create_dataloader_depth(args.train_split, True, args)
        dataloader_valid = create_dataloader_depth('valid', True, args)
        dataloader_test  = create_dataloader_depth('test', True, args)
    else:
        dataloader_train = create_dataloader(args.train_split, True, args)
        dataloader_valid = create_dataloader('valid', True, args)
        dataloader_test  = create_dataloader('test', True, args)
    print('%d batches of training examples' % len(dataloader_train))
    print('%d batches of validation examples' % len(dataloader_valid))
    print('%d batches of testing examples' % len(dataloader_test))

    phrase_encoder = RecurrentPhraseEncoder(300, 300)


    if args.depth_on == 'on':
        model = DRNet_depth(phrase_encoder, args.feature_dim, args.num_layers,args)
    else:
        model = DRNet(phrase_encoder, args.feature_dim, args.num_layers,args)
    
    model.cuda()
    

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.l2)
    if args.train_split == 'train':
        scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
    else:
        scheduler = StepLR(optimizer, step_size=args.patience, gamma=0.1) 

    start_epoch = 0
    if args.resume != None:
        print(' => loading model checkpoint from %s..' % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    best_acc = -1.

    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        print('epoch #%d' % epoch)
   
        print('training..')
        loss, acc = train(model, criterion, optimizer, dataloader_train, epoch, args)
        print('\n\ttraining loss = %.4f' % loss)
        print('\ttraining accuracy = %.3f' % acc)
        checkpoint_filename = os.path.join(args.log_dir, 'checkpoints/model_%.3f_%02d.pth' % (acc,epoch))
        if epoch % 5 == 0:
            print('validating..')
            torch.cuda.synchronize()
            start = time.time()
            loss, acc = test('valid', model, criterion, dataloader_valid, epoch, args)
            torch.cuda.synchronize()
            end = time.time()
            dtime = ((end - start)/len(dataloader_valid)/args.batchsize)
            print('\n\tvalidation loss = %.4f' % loss)
            print('\tvalidation accuracy = %.3f' % acc)
            print('\tvalidation time per input = %.3f' % dtime )


        
        model.cpu()
        torch.save({'epoch': epoch + 1,
                    'args': args,
                    'state_dict': model.state_dict(),
                    'accuracy': acc,
                    'optimizer' : optimizer.state_dict(),
                   }, checkpoint_filename)
        model.cuda()
     
        if args.train_split != 'train_valid' and best_acc < acc:
            best_acc = acc
            shutil.copyfile(checkpoint_filename, os.path.join(args.log_dir, 'checkpoints/model_best.pth'))
            shutil.copyfile(os.path.join(args.log_dir, 'predictions/pred_%02d.pickle' % epoch), 
                            os.path.join(args.log_dir, 'predictions/pred_best.pickle'))

        if args.train_split == 'train':
            scheduler.step(loss)
        else:
            scheduler.step()

    print('testing..')
    loss, acc = test('test', model, criterion, dataloader_test, None, args)
    print('\n\ttesting loss = %.4f' % loss)
    print('\ttesting accuracy = %.3f' % acc)
    fh.close()
    


if __name__ == '__main__':
    main()
