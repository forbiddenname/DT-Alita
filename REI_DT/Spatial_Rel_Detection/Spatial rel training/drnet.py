import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from models.hourglass import Hourglass


class DRNet(nn.Module):

    def __init__(self, phrase_encoder, feature_dim, num_layers, args,backbone='resnet18'):
        super(DRNet, self).__init__()
    
        self.phrase_encoder = phrase_encoder
        self.num_layers = num_layers

        if args.apprfeat_on == 'on':
            self.appr_module = models.__dict__[backbone](pretrained=True)
            self.appr_module.fc = nn.Linear(512, 256) 

        if args.spatial_feat == 'multifeat':
            self.linear2 = nn.Linear(8, 256)
            self.batchnorm2 = nn.BatchNorm1d(256)
        elif args.spatial_feat == 'dualmask':
            self.pos_module = nn.Sequential(OrderedDict([
                ('conv1_p', nn.Conv2d(2, 32, 5, 2, 2)),
                ('batchnorm1_p', nn.BatchNorm2d(32)),
                ('relu1_p', nn.ReLU()),
                ('conv2_p', nn.Conv2d(32, 64, 3, 1, 1)),
                ('batchnorm2_p', nn.BatchNorm2d(64)),
                ('relu2_p', nn.ReLU()),
                ('maxpool2_p', nn.MaxPool2d(2)),
                ('hg', Hourglass(8, 64)), 
                ('batchnorm_p', nn.BatchNorm2d(64)),
                ('relu_p', nn.ReLU()),
                ('maxpool_p', nn.MaxPool2d(2)),
                ('conv3_p', nn.Conv2d(64, 256, 4)),
                ('batchnorm3_p', nn.BatchNorm2d(256)),
            ]))

        if args.apprfeat_on =='on' and args.spatial_feat !='off':
            self.PhiR_0 = nn.Linear(512, feature_dim)
        elif args.apprfeat_on =='on' and args.spatial_feat == 'off':
            self.PhiR_0 = nn.Linear(256, feature_dim)
        elif args.apprfeat_on =='off' and args.spatial_feat != 'off':
            self.PhiR_0 = nn.Linear(256, feature_dim)

        self.batchnorm = nn.BatchNorm1d(feature_dim)   #数据归一化

        self.PhiA = nn.Linear(300, feature_dim)  ##### 300 X feature_dim
        self.PhiB = nn.Linear(300, feature_dim)  ##### 300 X feature_dim
        self.PhiR = nn.Linear(feature_dim, feature_dim)
 
        self.fc = nn.Linear(feature_dim, args.catelen)


    # def forward(self, subj, obj, im, posdata, predicates):
    #     appr_feature = self.appr_module(im)

    #     pos_feature = self.pos_module(posdata)
    #     if pos_feature.size(0) == 1:
    #         pos_feature = torch.unsqueeze(torch.squeeze(pos_feature), 0)
    #     else:
    #         pos_feature = torch.squeeze(pos_feature)

    #     qr = F.relu(self.batchnorm(self.PhiR_0(torch.cat([appr_feature, pos_feature], 1))))

    #     qa = self.phrase_encoder(subj)
    #     qb = self.phrase_encoder(obj)
    #     for i in range(self.num_layers):
    #         qr = F.relu(self.PhiA(qa) + self.PhiB(qb) + self.PhiR(qr))
    
    #     qr = self.fc(qr)

    #     return torch.sum(qr * predicates, 1)


    def forward(self, subj, obj, im, posdata, multispa,args):
    # def forward(self, subj, obj, im, posdata, predicates): 
        if args.apprfeat_on == 'on':
            appr_feature = self.appr_module(im)   # batchsize X 256

        if args.spatial_feat == 'dualmask':
            pos_feature = self.pos_module(posdata)
            if pos_feature.size(0) == 1:
                pos_feature = torch.unsqueeze(torch.squeeze(pos_feature), 0)
            else:
                pos_feature = torch.squeeze(pos_feature)     # batchsize X 256
        elif args.spatial_feat == 'multifeat':
            # print(multispa.size())
            pos_feature = self.linear2(multispa)
            pos_feature = self.batchnorm2(pos_feature)
            pos_feature = F.relu(pos_feature)

        if args.apprfeat_on =='on' and args.spatial_feat !='off':
            qr = F.relu(self.batchnorm(self.PhiR_0(torch.cat([appr_feature, pos_feature], 1))))
        elif args.apprfeat_on =='on' and args.spatial_feat == 'off':
            qr = F.relu(self.batchnorm(self.PhiR_0(torch.cat([appr_feature], 1))))
        elif args.apprfeat_on =='off' and args.spatial_feat != 'off':
            qr = F.relu(self.batchnorm(self.PhiR_0(torch.cat([pos_feature], 1))))


        

        qa = self.phrase_encoder(subj)    # 10 X 300
        qb = self.phrase_encoder(obj)     # 10 X 300
        for i in range(self.num_layers):
            qr = F.relu(self.PhiA(qa) + self.PhiB(qb) + self.PhiR(qr)) ### batchsize X 512 + batchsize X 512 + batchsize X 512 = batchsize X 512
        qr = self.fc(qr)
        # print(qr)
        # print(predicates)
        return qr
        #return torch.sum(qr * predicates, 1)