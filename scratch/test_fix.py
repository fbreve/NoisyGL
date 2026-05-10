import sys
import os
import numpy as np

# Add GCN+LNPCC to path
sys.path.insert(0, os.path.abspath('../GCN+LNPCC'))

from lnpcc import LabelNoisePCC

def test_isolated_node():
    print("Testing realistic small scenario (potential double-free)...")
    # 4 nodes, 2 edges
    neib_list = np.array([
        [1, -1],
        [0, -1],
        [-1, -1], # Isolated
        [-1, -1]  # Isolated labeled node?
    ], dtype=np.int64)
    neib_qt = np.array([1, 1, 0, 0], dtype=np.int64)
    
    # 2 classes
    labels = np.array([0, 1, -1, -1], dtype=np.int64)
    
    model = LabelNoisePCC(impl="cython")
    model.set_graph(neib_list, neib_qt)
    
    try:
        # We need at least 10 repeats or a high max_iter to trigger many steps
        res = model.fit_predict(labels, n_repeats=2, max_iter=100)
        print("Success: fit_predict finished without crash.")
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    test_isolated_node()
