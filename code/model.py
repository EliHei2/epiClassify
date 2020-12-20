import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,n_features,n_layers,n_classes):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(n_features,n_features))
        self.last_layer = nn.Linear(n_features,n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self,n_features,n_classes,n_hidden=4):
        super(GNN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        #self.conv1 = GCNConv(1, n_hidden)
        #self.conv1 = pyg_nn.GINConv(nn.Sequential(nn.Linear(1,n_hidden),nn.ReLU(), nn.Linear(n_hidden, n_hidden)))

        #Use GraphSAGE layer:
        #https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
        self.conv1 = pyg_nn.SAGEConv(1,n_hidden)
        #self.conv2 = GCNConv(n_hidden,n_hidden)
        self.linear = nn.Linear(n_features*n_hidden,n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        
        # Resize from (batch_size * n_features, n_hidden) to (batch_size, n_features * n_hidden)
        x = x.view(-1,self.n_features*self.n_hidden)
        x = self.linear(x)

        return x
