import argparse
from itertools import product

from asap import ASAP
from datasets import get_dataset
from diff_pool import DiffPool
from edge_pool import EdgePool
from gcn import GCN, GCNWithJK
from gin import GIN, GIN0, GIN0WithJK, GINWithJK
from global_attention import GlobalAttentionNet
from graclus import Graclus
from graph_sage import GraphSAGE, GraphSAGEWithJK
from sag_pool import SAGPool
from set2set import Set2SetNet
from sort_pool import SortPool
from top_k import TopK
from mixup import MixUPNet
from train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--size_strat', action='store_true')
args = parser.parse_args()

# layers = [1, 2, 3, 4, 5]
# hiddens = [16, 32, 64, 128]
layers = [2]
hiddens = [64]
datasets = [
    "DD",
    "PROTEINS",
    "FRANKENSTEIN",
    "PTC_MR",
    'IMDB-BINARY'
    'COLLAB',
    'obgb-molhiv'
]  # , 'COLLAB']


nets = [
    # GCNWithJK,
    # GraphSAGEWithJK,
    # GIN0WithJK,
    # GINWithJK,
    # Graclus,
    GCN,
    TopK,
    # SAGPool,
    DiffPool,
    EdgePool,
    # GCN,
    GraphSAGE,
    # GIN0,
    GIN,
    # GlobalAttentionNet,
    # Set2SetNet,
    MixUPNet,
    SortPool,
    # ASAP,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    head_acc, med_acc, tail_acc = info["head_acc"], info["med_acc"], info["tail_acc"]
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f},'
          f'Test Accuracy: {test_acc:.3f}, '
          f'Head Accuracy: {head_acc:.3f}, ',
          f'Med  Accuracy: {med_acc:.3f}, ',
          f'Tail Accuracy: {tail_acc:.3f}')


results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
        loss, acc, std, head_acc, head_std, med_acc, med_std, tail_acc, tail_std = cross_validation_with_val_set(
            dataset,
            model,
            folds=5,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=logger,
            size_strat=args.size_strat
        )
        with open("metrics.txt", "a") as txt_file:
            txt_file.write(f"Dataset: {dataset_name}, \n"
                           f"Model: {Net.__name__}, \n"
                           f"Num Layers: {num_layers}, \n"
                           f"Hidden Dim: {hidden}, \n"
                           f"Test Mean: {round(acc, 4)}, \n"
                           f"Std Test Mean: {round(std, 4)}, \n"
                           f"Head Mean: {round(head_acc, 4)}, \n"
                           f"Std Head Mean: {round(head_std, 4)}, \n"
                           f"Medium Mean: {round(med_acc, 4)}, \n"
                           f"Std Medium Mean: {round(med_std, 4)}, \n"
                           f"Tail Mean: {round(tail_acc, 4)}, \n"
                           f"Std Tail Mean: {round(tail_std, 4)} \n\n"
                           )
        if loss < best_result[0]:
            best_result = (loss, acc, std, head_acc, head_std, med_acc, med_std, tail_acc, tail_std)

    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']
results = '\n'.join(results)
print(f'--\n{results}')