import copy
import os.path as osp

import torch
import torch_geometric.data

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset


class NormalizedDegree:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def add_org_nodes(data: torch_geometric.data.Data):
    data.org_nodes = torch.tensor(data.num_nodes, dtype=torch.long)
    data.x = data.x.to(torch.float32)
    return data


def check_ogb(name, path, cleaned=False):
    if name.startswith('ogbg'):
        if not name.endswith('molhiv'):
            raise ValueError(f"Invalid dataset - {name}, which doesn't support classification")
        dataset = PygGraphPropPredDataset(name=name, root=path, transform=add_org_nodes)
    else:
        dataset = TUDataset(path, name, transform=add_org_nodes, cleaned=cleaned)
    return dataset
        

def convert_dataset(dataset, out_path):
    s= f"{len(dataset)}\n"

    for g_idx in tqdm(range(len(dataset))):
      s += f'{dataset[g_idx].num_nodes} {dataset[g_idx].y.item()}\n'
      for n_idx in range(dataset[g_idx].num_nodes):
        if hasattr(dataset, 'x') and dataset.x.shape[-1] != 0:
          tag = dataset[g_idx].x[n_idx].nonzero().item()
        else:
          tag = 0
        edges = dataset[g_idx].edge_index[1][dataset[g_idx].edge_index[0] == n_idx].tolist()
        s += f"{tag} {len(edges)} {' '.join([str(edge) for edge in edges])}\n"
    
    with open(out_path, 'w') as txt_file:
        txt_file.write(txt_file)


def get_dataset(name, sparse=True, cleaned=False, save_data=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = check_ogb(name, path)

    if isinstance(dataset, PygGraphPropPredDataset):
        dataset.data.edge_attr = None
        print(f"[INFO] {name} dataset selected.")
        return dataset

    dataset.data.edge_attr = None

    if dataset.data.x is None or dataset.data.x.shape[1] == 0:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            if dataset.transform is not None:
                dataset.transform = T.Compose(
                    [dataset.transform, T.OneHotDegree(max_degree)])
            else:
                dataset.transform = T.OneHotDegree(max_degree)

        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            if dataset.transform is not None:
                dataset.transform = T.Compose(
                    [dataset.transform, NormalizedDegree(mean, std)])
            else:
                dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)

        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    if save_data:
        os.makedirs(os.path.join("dataset_transformed", name), exist_ok=True)
        convert_dataset(dataset, os.path.join('dataset_trasnformed', name, f'{name}.txt')))

    return dataset
