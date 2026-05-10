import sys
import os
import numpy as np

# Add paths to sys.path
sys.path.insert(0, os.path.abspath('../GCN+LNPCC'))
sys.path.insert(0, os.path.abspath('../pypcc'))

try:
    from lnpcc import LabelNoisePCC
    print("[SUCCESS] lnpcc loaded")
    
    # Simple test run for lnpcc
    n_nodes = 100
    n_feat = 10
    X = np.random.rand(n_nodes, n_feat)
    y = np.random.randint(0, 2, n_nodes)
    y[50:] = -1 # some unlabeled
    
    lnpcc = LabelNoisePCC()
    lnpcc.build_graph(X, y, k_nn=5)
    res = lnpcc.fit_predict(y, max_iter=100)
    print(f"[SUCCESS] lnpcc test run finished. Result size: {len(res)}")

except Exception as e:
    print(f"[FAILURE] lnpcc error: {e}")
    import traceback
    traceback.print_exc()

try:
    from pcc import ParticleCompetitionAndCooperation
    print("[SUCCESS] pypcc loaded")
    
    # Simple test run for pypcc
    pcc = ParticleCompetitionAndCooperation()
    pcc.build_graph(X, k_nn=5)
    res = pcc.fit_predict(y, max_iter=100)
    print(f"[SUCCESS] pypcc test run finished. Result size: {len(res)}")

except Exception as e:
    print(f"[FAILURE] pypcc error: {e}")
    import traceback
    traceback.print_exc()
