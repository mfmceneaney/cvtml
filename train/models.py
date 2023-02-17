#----------------------------------------------------------------------#
# Author: M. McEneaney, Duke University
# Date: Feb. 2023
#----------------------------------------------------------------------#

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(in_channels, hidden_channels).jittable() #NOTE: NEEDED FOR DEPLOYMENT IN CMAKE
        self.conv2 = GraphConv(hidden_channels, hidden_channels).jittable()
        self.conv3 = GraphConv(hidden_channels, hidden_channels).jittable()
#         self.lin1 = Linear(hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin1(x)
#         x = self.lin2(x)
        x = self.lin3(x)
        x = torch.sigmoid(x) #NOTE: DON'T SOFTMAX IF USING BCELOSS, USE SIGMOID INSTEAD
        
        return x
