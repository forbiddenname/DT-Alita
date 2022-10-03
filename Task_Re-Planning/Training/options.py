import os
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/home/lx/TaskPlaning/datasets/dataset_train')
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--trainratio', type=float, default=0.8)
    
    parser.add_argument('--exp_id', type=str, default='gcnlstm')
    parser.add_argument('--log_dir', type=str, default='./runs')
    # parser.add_argument('--log_dir', type=str, default=os.path.join('./runs', str(datetime.now())[:-7]))
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--seq_len', type=int, default=15, help = 'lstm最大步数')

    parser.add_argument('--num_workers', type=int, default=5)   #工作进程数量,batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM
    parser.add_argument('--resume', type=str, help='model checkpoint to resume')
    # parser.add_argument('--train_split', type=str, default='train', choices=['train', 'train_valid'])
    # parser.add_argument('--pretrained', action='store_true')
    # parser.add_argument('--depth_on', type=str, default = 'on')
    parser.add_argument('--lstmdrop', type=float, default= 0.)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--visual_feature_size', type=int, default=3)
    # parser.add_argument('--predicate_embedding_dim', type=int, default=512)

    parser.add_argument('--hidden_size', type=int, default=1280, help = 'lstm的hidden size')

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512)

    parser.add_argument('--cksave', type=str, default= 'false', help = '是否存储checkpoint')
    parser.add_argument('--stiff', type=str, default='off',help = '刚度？')
    parser.add_argument('--rel', type=str, default='on',help = '物体间关系？')
    parser.add_argument('--label', type=str, default='on',help = '实体标签？')
    parser.add_argument('--task_encode', type=str, default='word2vec',help = '任务编码方式')

    parser.add_argument('--model', type=str, choices=['gcnlstm', 'mlplstm','mlp','nn','gcnmlp'], default='gcnlstm')



    args = parser.parse_args()

    args.object_raw_categories = ['manipulator',
                        'sponge',
                        'cup',
                        'bowl',
                        'plate',
                        'apple',
                        'banana',
                        'orange',
                        'kettle',
                        'lid'
                        ]
    args.predicate_categories = ['nothing',
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

    # args.actions_categories = ['start',
    #                     'stop',
    #                     'approach',
    #                     'grasp',
    #                     'movetogether',
    #                     'place',
    #                     'pour',
    #                     'push',
    #                     'stir '
    #                 ]

    # args.objects_categories = ['start',
    #                     'stop',
    #                     'sponge',
    #                     'cup',
    #                     'bowl',
    #                     'plate',
    #                     'banana',
    #                     'apple',
    #                     'orange'
    #                     ]

    # args.task_categories = ['clean up the desktop',
    #                 'pour water from cup',
    #                 'put green apple into bowl']


    args.actions_categories = ['start',
                        'stop',
                        'approach',
                        'grasp',
                        'movetogether',
                        'place',
                        'pourinto',
                        'push',
                        'stir ',
                        'hold',
                        'nothing'
                    ]

    args.objects_categories = ['start',
                        'stop',
                        'sponge',
                        'cup',
                        'bowl',
                        'plate',
                        'banana',
                        'apple',
                        'orange',
                        'kettle',
                        'lid'
                        ]
    args.task_categories = ['clean up the desktop', ## clean up the desktop
                    'get a cup of water', ## get a cup of water
                    'wash the bowl', ## wash the bowl
                    'put banana onto plate', ## put an apple on plate
                    'pour water from cup into bowl', ## Pour the water from the cup into the bowl
                    'wash the green fruits']


    # args.max_phrase_len = 2
    # args.node_spalenth = 7 if args.stiff == 'off' else 8
    args.node_spalenth = 7
    args.actobjlen = len(args.actions_categories)+len(args.objects_categories)
    maxtasklen = 0
    for f1 in args.task_categories:
        f = f1.split(' ')
        if len(f) > maxtasklen:
            maxtasklen = len(f)
    args.maxtasklen = maxtasklen
        

    if args.exp_id != None and args.cksave == 'true':
        args.log_dir = os.path.join('./runs', args.exp_id)
        args.log_dir = os.path.join(args.log_dir, str(datetime.now())[:-7])
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        os.makedirs(os.path.join(args.log_dir, 'predictions'))
        os.makedirs(os.path.join(args.log_dir, 'checkpoints')) 

    print(args)
    return args


if __name__ == '__main__':
    args = parse_args()
