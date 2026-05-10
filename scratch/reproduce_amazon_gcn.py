import sys
import os
# Add project root to path so we can import 'utils' and 'predictor'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torch.nn as nn
import threading

# Use the same environment settings we applied for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "none"

import argparse
from utils.dataloader import Dataset
from predictor.module.GNNs import GCN

def test_gcn(device_id, n_nodes, n_feat, n_classes):
    device = torch.device(f'cuda:{device_id}')
    print(f"[Worker] Testing GCN on {device}...")
    
    # Simulate a typical HPO config
    model = GCN(
        in_channels=n_feat,
        hidden_channels=128,
        out_channels=n_classes,
        n_layers=3,
        dropout=0.5
    ).to(device)
    
    # Dummy data
    feats = torch.randn(n_nodes, n_feat).to(device)
    edge_index = torch.randint(0, n_nodes, (2, 200000)).to(device)
    labels = torch.randint(0, n_classes, (n_nodes,)).to(device)
    mask = torch.ones(n_nodes, dtype=torch.bool).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"[Worker] Starting 10 simulated trials (200 steps each) on {device}...")
    for t_idx in range(10):
        print(f" [Trial {t_idx+1}/10] starting...")
        for i in range(200):
            optimizer.zero_grad()
            out = model(feats, edge_index)
            loss = torch.nn.functional.cross_entropy(out[mask], labels[mask])
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print(f"  Step {i+1} done on {device}")
    
    print(f"[Worker] Success: Finished all 10 trials on {device}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # amazon-ratings stats
    N_NODES = 24492
    N_FEAT = 300
    N_CLASSES = 5
    
    # Set stack size to 8MB like in our fix
    threading.stack_size(8 * 1024 * 1024)
    
    def run_in_thread():
        try:
            test_gcn(args.gpu, N_NODES, N_FEAT, N_CLASSES)
        except Exception as e:
            print(f"FAILED: {e}")

    t = threading.Thread(target=run_in_thread)
    t.start()
    t.join()
