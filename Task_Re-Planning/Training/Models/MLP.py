import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
# from .GN_Layers import Edge_node_Model
from .recurrent_phrase_encoder import RecurrentPhraseEncoder
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np


class MLP(nn.Module):
    def __init__(self,args):
        super(MLP, self).__init__()
        self.nodenum = len(args.object_raw_categories)
        self.batch_size = args.batchsize
        self.node_spalenth = args.node_spalenth
        self.actobjlen = args.actobjlen
        self.lstmdrop = args.lstmdrop
        self.phrase_encoder = RecurrentPhraseEncoder(300, 64)
        self.pos_encoder = Lin(self.node_spalenth, 64)
        self.actseq_encoder = Lin(self.actobjlen, 640)
        
        self.rel = args.rel
        if self.rel == 'on':
            self.rel_encoder = Lin(self.nodenum, 64)
            self.node_encoder = Seq(Lin(64*3, 64), ReLU())
        elif self.rel == 'off':
            self.node_encoder = Seq(Lin(64*2, 64), ReLU())
        self.taskfea_encode = Seq(Lin(300*args.maxtasklen,640), ReLU())

        self.fc1 = Seq(Lin(640*3, 1024), ReLU())        
        # self.lstm = nn.LSTM(self.actobjlen,args.hidden_size,args.num_layers,batch_first = True) ###(intput_size,hidden_size,num_layers)
        self.actfc = Lin(1024,len(args.actions_categories))
        self.objfc = Lin(1024,len(args.objects_categories))
        
    def forward(self, x_raw, task_fea_vec, action_seq, lengths, mode):
        #
        # x_raw     ： 场景图的节点特征(label+pos+rel)
        # edge_index： 场景图的边索引 
        # edge_attr ： 场景图的边特征
        # action_seq： 数据集中 <action, object>序列
        # mode      ： mode为1表示训练模式，为0表示eva模式
        #
        ######prepocessesing
        # print(x_raw.shape)
        x_raw = x_raw.reshape([-1,300+self.node_spalenth+self.nodenum])
        # print(x_raw.shape)
        x_label = self.phrase_encoder(torch.unsqueeze(x_raw[:,0:300],1))
        x_pos   = self.pos_encoder(x_raw[:,300:300+self.node_spalenth])
        if self.rel == 'on':
            x_rel = self.rel_encoder(x_raw[:,300+self.node_spalenth:x_raw.size(1)])
            x = torch.cat([x_label, x_pos, x_rel],1)
        elif self.rel == 'off':
            x = torch.cat([x_label, x_pos],1)

        nodex_ = self.node_encoder(x)
        graph_attr = nodex_.reshape(len(nodex_)//self.nodenum, self.nodenum*64)

        ######将每个batch的action_sequence按照大小排序，用于pack_padded
        
        # lengths = torch.sum(torch.sum(action_seq, dim = 2),dim = 1)//2       
        # a_lengths, idx = lengths.sort(dim = 0,descending = True)
        # _, un_idx = torch.sort(idx, dim=0)
        # action_seq_order = action_seq[idx]
        
#         print(lengths)
#         print(a_lengths)
#         print(idx)
#         print(un_idx)
        # graph_attr_order = graph_attr[idx]

        task_fea = self.taskfea_encode(task_fea_vec)   ###6×300 task feature To 512
        # task_fea_order = task_fea[idx]
        # print(graph_attr.shape)
        # print(task_fea.shape)
        # print(action_seq.shape)
        batchs = graph_attr.shape[0]
        actseq = action_seq.shape[1]
        fealen = graph_attr.shape[1]
        # print(fealen)
        graph_attr_1 = graph_attr.repeat([1,action_seq.shape[1]]).reshape([batchs,actseq,fealen])
        task_fea_1 = task_fea.repeat([1,action_seq.shape[1]]).reshape([batchs,actseq,fealen])
        act_fea_1 = self.actseq_encoder(action_seq)

        attr_all = torch.cat([graph_attr_1, task_fea_1,act_fea_1],2)
        # print(graph_attr_order.shape)
#         graph_attr_order = graph_attr_order + task_fea_order
#         print(graph_attr_order.shape)
#         print(graph_attr_order)
        
#         print(action_seq_order.shape)
#         print(graph_attr_order.shape)
        outputs = torch.tensor(np.zeros([batchs,actseq,1024]),dtype=torch.float32).cuda()
        if mode == 1 : ##train stage

            # inputs = action_seq_order.reshape(9,batch_size,14)  ###(seq_len,batch_size,input_size)
            # inputs = action_seq
            for i in range(0,actseq):
                outputs[:,i,:] = self.fc1(attr_all[:,i,:].squeeze(1))
            
            # print(outputs.shape)
            # inputs_pack = pack_padded_sequence(inputs,a_lengths,batch_first = True)
#             print(inputs_pack)
            # h0 = torch.unsqueeze(graph_attr_order,0)  ### (num_layers* 1,batch_size,hidden_size)
            # c0 = torch.zeros(h0.shape).cuda()   ### (num_layers* 1,batch_size,hidden_size)
            # outputs_packs, (hn, cn) = self.lstm(inputs_pack, (h0, c0))   ### (outpus: (batch_size,seq_len,hidden_size), hn: (num_layers*1,batch_size,hidden_size))
            # outputs, _ = pad_packed_sequence(outputs_packs,batch_first = True)
            # if self.lstmdrop > 0:
            #     outputs = F.dropout(outputs, p=self.lstmdrop, training=self.training)
            # action_seq_lstm_order = outputs.reshape(batch_size,9,1024)
            # action_seq_lstm_order = outputs
            # 根据un_idx将输出转回原输入顺序
            # action_seq_lstm = torch.index_select(action_seq_lstm_order,0,un_idx)
#             print(action_seq_lstm_order)
#             print(action_seq_lstm)
            action_seq = outputs.reshape([-1,1024])
            # action_seq = F.dropout(action_seq, p=0.1, training=self.training)
            # objetcs_batch = F.dropout(objetcs_batch, p=0.2, training=self.training)
            actions_batch = self.actfc(action_seq) ###(batch_size,seq_len,actions_categories)
            objetcs_batch = self.objfc(action_seq) ###(batch_size,seq_len,objects_categories)
            actions_batch = F.dropout(actions_batch, p=0.2, training=self.training)
            objetcs_batch = F.dropout(objetcs_batch, p=0.2, training=self.training)
#             print(actions_batch)
            # print(actions_batch.shape)
            # actions_batch_pre = F.softmax(actions_batch,dim=2)
            # objetcs_batch_pre = F.softmax(actions_batch,dim=2)
#             print(actions_batch_pre.shape)
            return actions_batch, objetcs_batch
        # elif mode == 0: ##test stage
            


        
