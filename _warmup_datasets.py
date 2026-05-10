"""
_warmup_datasets.py
Minimal script that triggers PyG dataset download/cache for one dataset.
Called by run_lnpcc_parallel.py before launching workers.

Usage: python _warmup_datasets.py <dataset_name>
"""
import sys
import os

# NoisyGL root must be the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataloader import pyg_load_dataset  # only downloads, no split needed

DS_PATH = './data'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: _warmup_datasets.py <dataset_name>")
        sys.exit(1)

    ds_name = sys.argv[1]
    try:
        pyg_load_dataset(ds_name, path=DS_PATH)
        print(f"[warmup] {ds_name}: done")
        sys.exit(0)
    except Exception as e:
        print(f"[warmup] {ds_name}: ERROR -- {e}")
        sys.exit(1)
