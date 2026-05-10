
import sys
import os
import numpy as np
import torch

# Add parent dirs to path
sys.path.append(os.path.abspath('c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL'))
sys.path.append(os.path.abspath('c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/GCN+LNPCC'))

from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed
from utils.labelnoise import label_process
from lnpcc_graph import build_graph_from_edge_index, build_knn_graph, build_augmented_graph

def check_degrees():
    dname = 'amazon-ratings'
    device = 'cpu'
    seed = 3000
    noise_type = 'pair'
    noise_rate = 0.5
    k_nn = 14

    print(f"Loading {dname}...")
    conf = load_conf('./config/_dataset/' + dname + '.yaml')
    data = Dataset(dname, path='./data/', 
                   feat_norm=conf.norm['feat_norm'], adj_norm=conf.norm['adj_norm'],
                   train_size=conf.split['train_size'], val_size=conf.split['val_size'], test_size=conf.split['test_size'],
                   train_percent=conf.split['train_percent'], val_percent=conf.split['val_percent'], test_percent=conf.split['test_percent'],
                   device=device)
    
    data.noisy_label, modified_mask = label_process(labels=data.labels, features=data.feats,
                                                    n_classes=data.n_classes,
                                                    noise_type=noise_type, noise_rate=noise_rate,
                                                    random_seed=seed)
    
    n_nodes = data.feats.shape[0]
    edge_index_np = data.edge_index.numpy()
    train_idx = np.asarray(data.train_masks[0]) # assumes list of masks, which Dataset usually has
    if isinstance(data.train_masks, list):
        # wait, Dataset in NoisyGL usually has train_masks as a single array or list
        train_idx = data.train_masks
    
    train_bool = np.zeros(n_nodes, dtype=bool)
    train_bool[train_idx] = True

    print("Building graphs...")
    neib_list_edge, neib_qt_edge = build_graph_from_edge_index(n_nodes, edge_index_np)
    
    # Pre-process feats like in lnpcc_Predictor
    feats_np = data.feats.numpy()
    std = feats_np.std(axis=0)
    std[std == 0] = 1.0
    feats_std = (feats_np - feats_np.mean(axis=0)) / std
    
    neib_list_knn, neib_qt_knn = build_knn_graph(feats_std, k_nn=k_nn)
    
    print("Building augmented graph (strategy='d')...")
    # Scenario lnpcc_amazon-ratings_pair_0.5 uses knn_mode='d' (Trial 1 log shows knn=d)
    nl, nq = build_augmented_graph(
        neib_list_edge, neib_qt_edge, neib_list_knn, neib_qt_knn,
        data.noisy_label.numpy(), train_bool, strategy='d',
    )
    
    print(f"Max degree: {nq.max()}")
    print(f"Nodes with degree > 1024: {(nq > 1024).sum()}")
    
    if nq.max() > 1024:
        print("WARNING: Max degree exceeds Cython buffer limit (1024)!")

if __name__ == "__main__":
    try:
        check_degrees()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
