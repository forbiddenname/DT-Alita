####如果要做节点特征的变换
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
import torch
import torch.nn.functional as F
class Edge_node_Model(torch.nn.Module):

    def __init__(self, edgef_dim, nodef_dim, hidden_dim, edge_last_dim, node_last_dim):
        super(Edge_node_Model, self).__init__()
        self.edgef_dim = edgef_dim
        self.nodef_dim = nodef_dim
        self.hidden_dim = hidden_dim
        self.edge_last_dim = edge_last_dim
        self.node_last_dim = node_last_dim
        
        self.edge_mlp1 = Seq(Lin(self.nodef_dim * 2 + self.edgef_dim, self.hidden_dim), ReLU(), Lin(self.hidden_dim, self.edge_last_dim)) #625：起始节点特征维数+末端节点特征维数+边特征维数

        self.node_mlp1 = Seq(Lin(self.nodef_dim + self.edgef_dim, self.hidden_dim), ReLU(), Lin(self.hidden_dim, self.nodef_dim + self.edgef_dim)) #318：起始节点特征维数+边特征维数

        self.node_mlp2 = Seq(Lin(self.nodef_dim * 2 + self.edgef_dim, self.hidden_dim), ReLU(), Lin(self.hidden_dim, self.node_last_dim))


        # self.edge_mlp1 = Seq(Lin(self.nodef_dim * 2 + self.edgef_dim, self.edge_last_dim)) #625：起始节点特征维数+末端节点特征维数+边特征维数

        # self.node_mlp1 = Seq(Lin(self.nodef_dim + self.edgef_dim, self.hidden_dim), ReLU()) #318：起始节点特征维数+边特征维数

        # self.node_mlp2 = Seq(Lin(self.nodef_dim + self.hidden_dim, self.node_last_dim), ReLU())
        

    def forward(self, x, edge_index, edge_attr):
        
        ## edge messaging
        # src, dest: [E, F_x], where E is the number of edges. src = x[row], row为当前batch中起始节点对应的序号list，总数为边数
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs. 图的特征
        # batch: [E] with max entry B - 1.

        
        row = edge_index[0]    #边起始节点
        col = edge_index[1]    #边的末端节点
        
        src = x[row]      #起始节点的特征
        dest = x[col]     #末端节点的特征

        out = torch.cat([src, dest, edge_attr], dim=1)  ## E x (2 * nodef_dim + edgef_dim)
        edge = self.edge_mlp1(out)
        # edge = F.dropout(edge, p=0.5, training=self.training)
        

        ## node messaging
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        out = torch.cat([x[row], edge_attr], dim=1)  #起始节点特征 + 边特征 （E X  N+E）
        out = self.node_mlp1(out)
        # out = F.dropout(out, p=0.5, training=self.training)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0)) ##x.size(0)为节点数，（N X ?与out同列数，这里设为N+E）
        # print(out) 

        out = torch.cat([x, out], dim=1) ## （N X N+N+E） 
        node = self.node_mlp2(out)
        # node = F.dropout(node, p=0.5, training=self.training)
        # node.reshape([len(x)//8, 8, 307])
        return node,edge