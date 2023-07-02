import copy
from torch_scatter import scatter
from torch_geometric.nn.conv import CGConv
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self, atom_fea_dim, edge_dim, node_dim=64, num_layers=3, h_dim=128, classification=False, num_class=2):
        super().__init__()

        self.atom_fea_dim = atom_fea_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.classification = classification

        self.embedding = nn.Linear(atom_fea_dim, node_dim)

        conv_layer = CGConv(node_dim, edge_dim, batch_norm=True)
        self.layers = nn.ModuleList([copy.deepcopy(conv_layer) for _ in range(num_layers)])

        if not self.classification:
            self.fc = nn.Linear(node_dim, h_dim)
            self.softplus = nn.Softplus()
            self.fc_out = nn.Linear(h_dim, 1)
        else:
            self.fc = nn.Linear(node_dim, h_dim)
            self.softplus = nn.Softplus()
            self.fc_out = nn.Linear(h_dim, num_class)
            self.dropout = nn.Dropout()

    def forward(self, data):
        x = self.embedding(data.x)

        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)

        x = scatter(x, data.batch, dim=0, reduce='mean')

        if not self.classification:
            x = self.softplus(x)
            x = self.fc(x)
            x = self.softplus(x)
            x = self.fc_out(x)
        else:
            x = self.softplus(x)
            x = self.fc(x)
            x = self.softplus(x)
            x = self.dropout(x)
            x = self.fc_out(x)
            x = F.log_softmax(x, dim=1)

        return x
