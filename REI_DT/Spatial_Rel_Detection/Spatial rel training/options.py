import os
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--datapath', type=str, default='/home/lx/DRNet/SpatialSense/data/annotations.json')
    # parser.add_argument('--imagepath', type=str, default='/home/lx/DRNet/SpatialSense/data/image')
    # parser.add_argument('--exp_id', type=str)
    parser.add_argument('--log_dir', type=str, default=os.path.join('./runs', str(datetime.now())[:-7]))
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--resume', type=str, help='model checkpoint to resume')
    parser.add_argument('--train_split', type=str, default='train_valid', choices=['train', 'train_valid'])
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--batchsize', type=int, default=60)   ####defalt = 60 /30(for custom dataset)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--visual_feature_size', type=int, default=3)
    parser.add_argument('--predicate_embedding_dim', type=int, default=512)

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=512)
  
    # parser.add_argument('--model', type=str, choices=['drnet', 'drnet_multispatial', 'drnet_nonspatial','drnet_depth_custom','drnet_custom', 'drnet_depth_custom_multispatial'])

    parser.add_argument('--custom_on',type=str,default='on',choices=['on', 'off'])
    parser.add_argument('--depth_on',type=str,default='on',choices=['on', 'off'])
    parser.add_argument('--apprfeat_on',type=str,default='on',choices=['on', 'off'])
    parser.add_argument('--spatial_feat',type=str,default='dualmask',choices=['off','dualmask', 'multifeat'])


    # VipCNN
    parser.add_argument('--roi_size', type=int, default=6)
    # VipCNN & PPR-FCN
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet34', 'resnet101'], default='resnet18')

    args = parser.parse_args()
    if args.custom_on == 'on':
        args.datapath = '/home/lx/DRNet/datasets/dataToAnno/annotations.json'
        args.imagepath = '/home/lx/DRNet/datasets/imgs'
        # args.batchsize = 30
        args.predicate_categories = [   'nothing',
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
                                        'hold' ###touch or hold
                                        ]
    else:
        args.datapath = '/home/lx/DRNet/SpatialSense/data/annotations.json'
        args.imagepath = '/home/lx/DRNet/SpatialSense/data/image'
        # args.batchsize = 30
        args.predicate_categories = ['above',
                                    'behind',
                                    'in',
                                    'in front of',
                                    'next to',
                                    'on',
                                    'to the left of',
                                    'to the right of',
                                    'under'] 
    args.catelen = len(args.predicate_categories)
    args.max_phrase_len = 2

    if args.custom_on != None:
        args.log_dir = os.path.join('./runs', 'custom'+args.custom_on+'_'+'depth'+args.depth_on+'_'+'apprfeat'+args.apprfeat_on+'_'+'spatial'+args.spatial_feat)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        os.makedirs(os.path.join(args.log_dir, 'predictions'))
        os.makedirs(os.path.join(args.log_dir, 'checkpoints'))  

    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()
