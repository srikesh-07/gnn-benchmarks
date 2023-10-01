import time

import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):

    val_losses, accs, head_accs, med_accs, tail_accs, durations = [], [], [], [], [], []

    if dataset.name == "PROTEINS":
        K = [0, 371, 742, 1113]
    elif dataset.name == "PTC_MR":
        K = [0, 115, 230, 344]
    elif dataset.name == "IMDB-BINARY":
        K = [0, 333, 666, 1000]
    elif dataset.name == "DD":
        K = [0, 393, 785, 1178]
    elif dataset.name == "FRANKENSTEIN":
        K = [0, 1445, 2890, 4337]

    nodes = list()
    for i in range(len(dataset)):
        nodes.append(dataset[i].num_nodes)

    nodes.sort(reverse=True)

    ranges = dict()

    ranges["head"] = list(set(nodes[K[0]:K[1]]))
    ranges["med"] = list(set(nodes[K[1]:K[2]]))
    ranges["tail"] = list(set(nodes[K[2]:K[3]]))

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # elif hasattr(torch.backends,
        #              'mps') and torch.backends.mps.is_available():
        #     try:
        #         import torch.mps
        #         torch.mps.synchronize()
        #     except ImportError:
        #         pass

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            temp_acc, temp_head, temp_med, temp_tail = eval_acc(model, test_loader, ranges)
            accs.append(temp_acc)
            head_accs.append(temp_head)
            med_accs.append(temp_med)
            tail_accs.append(temp_tail)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
                "head_acc": head_accs[-1],
                "med_acc": med_accs[-1],
                "tail_acc": tail_accs[-1]
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    head_acc, med_acc, tail_acc = tensor(head_accs), tensor(med_accs), tensor(tail_accs)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    head_acc, med_acc, tail_acc = head_acc.view(folds, epochs), med_acc.view(folds, epochs), tail_acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    head_acc = head_acc[torch.arange(folds, dtype=torch.long), argmin]
    med_acc = med_acc[torch.arange(folds, dtype=torch.long), argmin]
    tail_acc = tail_acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    head_acc_mean = head_acc.mean().item()
    head_acc_std = head_acc.std().item()
    med_acc_mean = med_acc.mean().item()
    med_acc_std = med_acc.std().item()
    tail_acc_mean = tail_acc.mean().item()
    tail_acc_std = tail_acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std, head_acc_mean, head_acc_std, med_acc_mean, med_acc_std, tail_acc_mean, \
        tail_acc_std


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, ranges):
    model.eval()
    graph_correct = {0: 0, 1: 0, 2: 0}
    total_graphs = {0: 0, 1: 0, 2: 0}

    correct = 0
    for data in loader:
        data = data.to(device)
        batch_data = data.to_data_list()
        with torch.no_grad():
            pred = model(data).max(1)[1]
        for idx, graph in enumerate(batch_data):
            num_nodes = graph.num_nodes
            if num_nodes in ranges["head"]:
                graph_group = 2
            elif num_nodes in ranges["med"]:
                graph_group = 1
            elif num_nodes in ranges["tail"]:
                graph_group = 0
            else:
                assert False
            if pred[idx].cpu().item() == data.y[idx].cpu().item():
                graph_correct[graph_group] += 1
            total_graphs[graph_group] += 1
        correct += pred.eq(data.y.view(-1)).sum().item()
    return (correct / len(loader.dataset),
            graph_correct[2] / total_graphs[2],
            graph_correct[1] / total_graphs[1],
            graph_correct[0] / total_graphs[0])


def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
