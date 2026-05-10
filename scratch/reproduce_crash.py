
import sys
import os
import time

# Force single-threading like in single_exp.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import faulthandler

# Enable fault handler to see the exact C call that crashes
print("[REPRO] Enabling faulthandler...")
faulthandler.enable()

# Add project root to path
sys.path.append(os.path.abspath('c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL'))

from predictor.LNPCC_Predictor import lnpcc_Predictor
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed

def reproduce_crash():
    dname = 'amazon-ratings'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    noise_type = 'uniform'
    noise_rate = 0.3
    seed = 3000
    
    print(f"[REPRO] Starting reproduction for {dname} on {device}")
    
    setup_seed(seed)
    data_conf = load_conf('./config/_dataset/' + dname + '.yaml')
    data = Dataset(dname, path='./data/', 
                   feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                   train_size=data_conf.split['train_size'], val_size=data_conf.split['val_size'], test_size=data_conf.split['test_size'],
                   train_percent=data_conf.split['train_percent'], val_percent=data_conf.split['val_percent'], test_percent=data_conf.split['test_percent'],
                   train_examples_per_class=data_conf.split['train_examples_per_class'],
                   val_examples_per_class=data_conf.split['val_examples_per_class'],
                   test_examples_per_class=data_conf.split['test_examples_per_class'],
                   add_self_loop=data_conf.modify['add_self_loop'],
                   from_npz=data_conf.modify['from_npz_largest_component'],
                   device=device,
                   split_type=data_conf.split['split_type']) 
    
    model_conf = load_conf(None, 'lnpcc', dname)
    # Populate necessary fields normally handled by total_exp_lnpcc.py
    model_conf.model['n_feat'] = data.dim_feats
    model_conf.model['n_classes'] = data.n_classes
    model_conf.model['knn_mode'] = 'u'
    model_conf.training['debug'] = True
    model_conf.training['n_epochs'] = 5 
    
    # Fake noisy labels like in the real experiment
    from utils.labelnoise import label_process
    data.noisy_label, modified_mask = label_process(labels=data.labels, features=data.feats,
                                                    n_classes=data.n_classes,
                                                    noise_type=noise_type, noise_rate=noise_rate,
                                                    random_seed=seed, debug=True)
    
    predictor = lnpcc_Predictor(model_conf, data, device)
    print("\n[REPRO] Calling predictor.train()...")
    try:
        predictor.train()
        print("\n[REPRO] SUCCESS! No crash occurred in this standalone run.")
    except Exception as e:
        print(f"\n[REPRO] CAUGHT EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_crash()
