import torch
from torch import Tensor
from typing import Optional
import numpy as np
from torch_geometric.nn import GraphConv, Linear, JumpingKnowledge, global_add_pool
import torch.nn.functional as F


class MixUp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: Tensor,
                lam: int,
                ptr: Optional[Tensor] = None) -> Tensor:
        if ptr is not None:
            ptr = ptr.tolist()
        else:
            ptr = [0, x.shape[0]]
        tracker = ptr[0]
        random_indices = torch.zeros(x.shape[0], dtype=torch.long)
        for idx, ptr in zip(range(x.shape[0]), ptr[1:]):
            node_indices = np.arange(tracker, ptr)
            np.random.shuffle(node_indices)
            random_indices[tracker: ptr] = torch.tensor(node_indices, dtype=torch.long)
            tracker = ptr
        out_x = (x * lam) + (1 - lam) * x[random_indices]
        return out_x


class MixUPNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden,):
        super(MixUPNet, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='sum')
        self.convs = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for _ in range(num_layers - 1)
        ])
        self.mixup = MixUp()
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        if self.training:
            lam = np.random.beta(4.0, 4.0)
        else:
            lam = 1
        x, edge_index, batch, ptr = data.x, data.edge_index, data.batch, data.ptr
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.mixup(x, lam, ptr)
        x = global_add_pool(x, batch=batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__