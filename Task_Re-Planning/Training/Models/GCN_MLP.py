import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from .GN_Layers import Edge_node_Model
from .recurrent_phrase_encoder import RecurrentPhraseEncoder
import numpy as np



class GCN_MLP(nn.Module):
    def __init__(self,args):
        super(GCN_MLP, self).__init__()
        self.nodenum = len(args.object_raw_categories)  ##图节点个数
        self.actobjlen = args.actobjlen
        # self.batch_size = args.batchsize
        self.lstmdrop = args.lstmdrop
        # self.phrase_encoder = RecurrentPhraseEncoder(300, 64)
        # self.pos_encoder = Lin(args.node_spalenth, 64)
        self.taskfea_encode = Seq(Lin(300*args.maxtasklen,640), ReLU())
#         self.gn1 = Edge_node_Model(11,128,256,64,256)  ## edgef_dim, nodef_dim, hidden_dim, edge_last_dim, node_last_dim
#         self.gn2 = Edge_node_Model(64,256,512,11,64)
        self.gn1 = Edge_node_Model(11,128,512,11,128)  ## edgef_dim, nodef_dim, hidden_dim, edge_last_dim, node_last_dim
        self.gn2 = Edge_node_Model(11,128,512,11,64)
        # self.lstm = nn.LSTM(args.actobjlen,args.hidden_size,args.num_layers,batch_first = True) ###(intput_size,hidden_size,num_layers)
        self.fc1 = Seq(Lin(640*3, 1024), ReLU())
        self.actfc = Lin(1024,len(args.actions_categories))
        self.objfc = Lin(1024,len(args.objects_categories))

        self.node_spalenth = args.node_spalenth
        self.rel = args.rel
        self.stiff = args.stiff
        self.label = args.label
        self.modelis = args.model
        self.actseq_encoder = Lin(self.actobjlen, 640)

        if self.label == 'off':
            self.pos_encoder = Lin(args.node_spalenth, 128)
        elif self.stiff == 'off':
            self.phrase_encoder = RecurrentPhraseEncoder(300, 128)
        else:
            self.phrase_encoder = RecurrentPhraseEncoder(300, 64)
            self.pos_encoder = Lin(args.node_spalenth, 64)


    def forward(self, x_raw, edge_index, edge_attr, task_fea_vec, action_seq, lengths, mode):
        #
        # x_raw     ： 场景图的节点特征
        # edge_index： 场景图的边索引 
        # edge_attr ： 场景图的边特征
        # action_seq： 数据集中 <action, object>序列
        # mode      ： mode为1表示训练模式，为0表示eva模式
        #
        ######prepocessesing
        
        # x_raw = x_raw.reshape([-1,300+self.node_spalenth+self.nodenum])
        # print(x_raw.shape)
        if self.label =='off':
            x_pos   = self.pos_encoder(x_raw[:,300:300+self.node_spalenth])
            x = x_pos
        elif self.stiff == 'off':
            x_label = self.phrase_encoder(torch.unsqueeze(x_raw[:,0:300],1))
            x = x_label
        else:
            x_pos   = self.pos_encoder(x_raw[:,300:300+self.node_spalenth])
            x_label = self.phrase_encoder(torch.unsqueeze(x_raw[:,0:300],1))
            x = torch.cat([x_label, x_pos],1)

        if self.rel =='off':
            edge_attr1 = torch.zeros(edge_attr.shape)
        else:
            edge_attr1 = edge_attr
        # print(edge_attr.shape)

        # x_ = self.phrase_encoder(torch.unsqueeze(x_raw[:,0:300],1))
        # _x = self.pos_encoder(x_raw[:,300:x_raw.size(1)])
        # x = torch.cat([x_, _x],1)

        ######将每个batch的action_sequence按照大小排序，用于pack_padded
        
        # lengths = torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2       
        # a_lengths, idx = lengths.sort(dim = 0,descending = True)
        # _, un_idx = torch.sort(idx, dim=0)
        # action_seq_order = action_seq[idx]
        
#         print(lengths)
#         print(a_lengths)
#         print(idx)
#         print(un_idx)
        ###########two layers GN   Graph features
        node_attr_1, edge_1 = self.gn1(x, edge_index, edge_attr1.cuda())
        node_attr_2, edge_2 = self.gn2(node_attr_1, edge_index, edge_1)
#         print(node_attr_1.shape)
#         print(node_attr_2.shape)
        graph_attr = node_attr_2.reshape(len(node_attr_2)//self.nodenum, self.nodenum*64) ### 节点类别 8 和 节点特征维数64
        # graph_attr_order = graph_attr[idx]

        task_fea = self.taskfea_encode(task_fea_vec)   ###6×300 task feature To 512
        # task_fea_order = task_fea[idx]

        # graph_attr = torch.cat([graph_attr, task_fea],1)
        
        batchs = graph_attr.shape[0]
        actseq = action_seq.shape[1]
        fealen = graph_attr.shape[1]
        # print(fealen)
        graph_attr_1 = graph_attr.repeat([1,action_seq.shape[1]]).reshape([batchs,actseq,fealen])
        task_fea_1 = task_fea.repeat([1,action_seq.shape[1]]).reshape([batchs,actseq,fealen])
        act_fea_1 = self.actseq_encoder(action_seq)

        attr_all = torch.cat([graph_attr_1, task_fea_1,act_fea_1],2)

        outputs = torch.tensor(np.zeros([batchs,actseq,1024]),dtype=torch.float32).cuda()
        # print(graph_attr_order.shape)
#         graph_attr_order = graph_attr_order + task_fea_order
#         print(graph_attr_order.shape)
#         print(graph_attr_order)
        
        # batch_size = len(action_seq_order)
#         print(action_seq_order.shape)
#         print(graph_attr_order.shape)
        
        if mode == 1 : ##train stage

            for i in range(0,actseq):
                outputs[:,i,:] = self.fc1(attr_all[:,i,:].squeeze(1))


            action_seq = outputs.reshape([-1,1024])
            actions_batch = self.actfc(action_seq) ###(batch_size,seq_len,actions_categories)
            objetcs_batch = self.objfc(action_seq) ###(batch_size,seq_len,objects_categories)
            actions_batch = F.dropout(actions_batch, p=0.1, training=self.training)
            objetcs_batch = F.dropout(objetcs_batch, p=0.1, training=self.training)
            return actions_batch, objetcs_batch

        # elif mode == 0: ##test stage
            


        
