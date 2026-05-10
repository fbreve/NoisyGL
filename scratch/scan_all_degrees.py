
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.abspath('c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL'))

from utils.dataloader import Dataset
from utils.tools import load_conf

def scan_degrees():
    datasets = ['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho', 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire']
    results = []
    
    for dname in datasets:
        try:
            print(f"Loading {dname}...")
            conf = load_conf('./config/_dataset/' + dname + '.yaml')
            data = Dataset(dname, path='./data/', 
                           feat_norm=False, adj_norm=False,
                           train_size=1, val_size=1, test_size=1,
                           device='cpu')
            
            # Reconstruct degree correctly from adj
            adj = data.adj.coalesce()
            indices = adj.indices()
            row = indices[0]
            
            # Using torch-geometric degree for consistency
            from torch_geometric.utils import degree
            degs = degree(row, data.n_nodes)
            max_d = int(degs.max())
            mean_d = float(degs.mean())
            
            print(f"  Max Degree: {max_d}, Mean Degree: {mean_d:.2f}")
            results.append((dname, max_d, mean_d))
        except Exception as e:
            print(f"  Error loading {dname}: {e}")
            
    print("\nSummary:")
    for d, mx, mn in results:
        print(f"{d:20} | Max Degree: {mx:5} | Mean Degree: {mn:6.2f}")

if __name__ == "__main__":
    scan_degrees()
