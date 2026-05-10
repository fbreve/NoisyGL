import os
import faulthandler
faulthandler.enable()

# Force single-threading for all major math libraries to prevent Access Violations (0xC0000005) 
# and Stack Overruns (0xC0000409) on Windows during parallel trials.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Scikit-learn / Joblib / Loky specific hardening
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Specific Intel MKL/OpenMP stability settings for Windows
os.environ["KMP_AFFINITY"] = "none"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_SETTINGS"] = "0"

# Silence "expandable_segments not supported on this platform" warning on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import argparse
import torch
import warnings

# Ignore the "Converting sparse tensor to CSR format" warning from PyG
# as we will fix it where possible, but some internal PyG calls might still trigger it.
warnings.filterwarnings("ignore", message=".*Converting sparse tensor to CSR format.*")

torch.set_num_threads(1)
try:
    import nni
    _NNI_ACTIVE = nni.get_trial_id() != 'STANDALONE'
except Exception:
    nni = None
    _NNI_ACTIVE = False

from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.labelnoise import label_process
from predictor.NRGNN_Predictor import nrgnn_Predictor
from predictor.CP_Predictor import cp_Predictor
from predictor.Smodel_Predictor import smodel_Predictor
from predictor.Coteaching_Predictor import coteaching_Predictor
from predictor.GCN_Predictor import gcn_Predictor
from predictor.RTGNN_Predictor import rtgnn_Predictor
from predictor.CLNode_Predictor import clnode_Predictor
from predictor.RNCGLN_Predictor import rncgln_Predictor
from predictor.PIGNN_Predictor import pignn_Predictor
from predictor.GIN_Predictor import gin_Predictor
from predictor.DGNN_Predictor import dgnn_Predictor
from predictor.UnionNET_Predictor import unionnet_Predictor
from predictor.CGNN_Predictor import cgnn_Predictor
from predictor.JoCoR_Predictor import jocor_Predictor
from predictor.CRGNN_Predictor import crgnn_Predictor
from predictor.APL_Predictor import apl_Predictor
from predictor.SCE_Predictor import sce_Predictor
from predictor.Forward_Predictor import forward_Predictor
from predictor.Backward_Predictor import backward_Predictor
from predictor.LCAT_Predictor import lcat_Predictor
from predictor.R2LP_Predictor import r2lp_Predictor
from predictor.MLP_Predictor import mlp_Predictor
from predictor.TSS_Predictor import tss_Predictor

# GCN + LN-PCC unified predictor
from predictor.LNPCC_Predictor import lnpcc_Predictor


def set_priority(level):
    """Set the priority of the current process (Windows)."""
    if level == 'normal':
        return
    import psutil
    import os
    try:
        p = psutil.Process(os.getpid())
        if level == 'below_normal':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif level == 'idle':
            p.nice(psutil.IDLE_PRIORITY_CLASS)
        print(f'[priority] Process priority set to {level.upper()}', flush=True)
    except Exception as e:
        print(f'[priority] Warning: could not set priority: {e}', flush=True)


def flatten_nni_params(params):
    """Flatten NNI-style nested params {knn_mode: {_name: 's', k: 5}} → {knn_mode: 's', k: 5}.
    Also handles plain flat dicts produced by Optuna."""
    flat = {}
    if not params: return flat
    for k, v in params.items():
        if isinstance(v, dict) and '_name' in v:
            flat[k] = v['_name']
            for sub_k, sub_v in v.items():
                if sub_k != '_name':
                    flat[sub_k] = sub_v
        else:
            flat[k] = v
    return flat

def _apply_flat_params(model_conf, flat_params):
    """Write flat param dict into model_conf.model / model_conf.training."""
    for item, val in flat_params.items():
        if item in ['lr', 'weight_decay']:
            model_conf.training[item] = val
        else:
            model_conf.model[item] = val

def merge_params(model_conf):
    """Read params from NNI tuner and apply to model_conf."""
    raw_params = nni.get_next_parameter()
    print("Raw tuner params:", raw_params)
    turner_params = flatten_nni_params(raw_params)
    print("Flat params:", turner_params)
    _apply_flat_params(model_conf, turner_params)
    print(model_conf)
    return model_conf, turner_params


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='dblp',
                    choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                             'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                    help='Select dataset')
parser.add_argument('--method', type=str,
                    default='gcn',
                    choices=[
                        'gcn', 'lnpcc',
                        'gin', 'smodel', 'jocor', 'coteaching',
                        'apl', 'sce', 'forward', 'backward', 'lcat', 'mlp',
                        'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'tss',
                        'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn', 'r2lp',
                    ],
                    help='Select method')
parser.add_argument('--noise_type', type=str,
                    default='instance',
                    choices=['clean', 'uniform', 'pair', 'random', 'instance'],
                    help='Type of label noise')
parser.add_argument('--noise_rate', type=float,
                    default='0.3',
                    help='Label noise rate')
parser.add_argument('--device', type=str,
                    default='cuda:0',
                    help='Device')
parser.add_argument('--seed', type=int,
                    default=3000,
                    help='Random Seed')
parser.add_argument('--params_json', type=str, default=None,
                    help='JSON string of parameters to override (for retest)')
parser.add_argument('--priority', type=str, default='normal',
                    choices=['normal', 'below_normal', 'idle'],
                    help='Process priority (default: normal)')
args = parser.parse_args()


if __name__ == '__main__':
    set_priority(args.priority)
    print(args)
    data_path = './data/'
    data_conf = load_conf('./config/_dataset/' + args.dataset + '.yaml')
    if not _NNI_ACTIVE:
        setup_seed(args.seed)
    data = Dataset(args.dataset, path=data_path,
                   feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                   train_size=data_conf.split['train_size'],
                   val_size=data_conf.split['val_size'],
                   test_size=data_conf.split['test_size'],
                   train_percent=data_conf.split['train_percent'],
                   val_percent=data_conf.split['val_percent'],
                   test_percent=data_conf.split['test_percent'],
                   train_examples_per_class=data_conf.split['train_examples_per_class'],
                   val_examples_per_class=data_conf.split['val_examples_per_class'],
                   test_examples_per_class=data_conf.split['test_examples_per_class'],
                   add_self_loop=data_conf.modify['add_self_loop'],
                   from_npz=data_conf.modify['from_npz_largest_component'],
                   device=args.device,
                   split_type=data_conf.split['split_type'])
    print('Current device: ' + str(data.feats.device))
    model_conf = load_conf(None, args.method, data.name)
    nni_params = {}
    if _NNI_ACTIVE and nni is not None:
        model_conf, nni_params = merge_params(model_conf)
    elif args.params_json:
        import json
        try:
            nni_params = json.loads(args.params_json)
            flat_params = flatten_nni_params(nni_params)
            print(f"[Params] Overriding from --params_json: {flat_params}")
            _apply_flat_params(model_conf, flat_params)
            nni_params = flat_params  # keep flat for logging
        except Exception as e:
            print(f"[Params] Error parsing params_json: {e}")

    # ── Windows Stability: Move Predictor to High-Stack Thread ──
    # Increase stack size to 32MB for extremely dense graphs
    import threading
    threading.stack_size(32 * 1024 * 1024)
    
    result_container = {}
    def run_train_thread():
        try:
            # ── 1. Label Processing (Heavy graph-based perturbation) ──
            data.noisy_label, modified_mask = label_process(labels=data.labels, features=data.feats,
                                                            n_classes=data.n_classes,
                                                            noise_type=args.noise_type, noise_rate=args.noise_rate,
                                                            random_seed=args.seed, debug=True)

            model_conf.model['n_feat'] = data.dim_feats
            model_conf.model['n_classes'] = data.n_classes
            model_conf.training['debug'] = True

            # ── 2. Predictor Initialization (Triggers graph extraction) ──
            if args.method == 'lnpcc':
                predictor = lnpcc_Predictor(model_conf, data, args.device)
            elif args.method == 'gcn':
                predictor = gcn_Predictor(model_conf, data, args.device)
            else:
                predictor = eval(args.method + '_Predictor')(model_conf, data, args.device)
            
            predictor.modified_mask = modified_mask

            # ── 3. Training ──
            res = predictor.train()
            
            # Pack results and predictor (for timing access) into container
            result_container['result'] = res
            result_container['predictor_state'] = {
                '_t_pcc': getattr(predictor, '_t_pcc', 0.0),
                '_t_gcn': getattr(predictor, '_t_gcn', 0.0),
                'test_acc': res['test'] if isinstance(res, dict) else 0.0
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            result_container['error'] = e

    import time, csv
    t0_wall = time.time()

    # ── GPU timing via CUDA events (captures only GCN phase) ──────────────────
    _use_cuda = 'cuda' in args.device
    if _use_cuda:
        import torch
        _ev_start = torch.cuda.Event(enable_timing=True)
        _ev_end   = torch.cuda.Event(enable_timing=True)
        _ev_start.record()

    train_thread = threading.Thread(target=run_train_thread)
    train_thread.start()
    train_thread.join()

    if 'error' in result_container:
        print(f"FAILED during threaded execution: {result_container['error']}", flush=True)
        # Fallback exit with error code
        import sys
        sys.exit(1)

    result = result_container.get('result')
    p_state = result_container.get('predictor_state', {})
    print(f"FINAL_RESULT: {result}")

    t_cpu_pcc = p_state.get('_t_pcc', 0.0)
    t_gpu_via_predictor = p_state.get('_t_gcn', 0.0)

    if _use_cuda:
        import torch
        _ev_end.record()
        torch.cuda.synchronize()
        t_gpu_coarse = _ev_start.elapsed_time(_ev_end) / 1000.0
    else:
        t_gpu_coarse = 0.0

    # If the predictor already measured the GCN phase (e.g., lnpcc), use it.
    if t_gpu_via_predictor > 0:
        t_gpu_gcn = t_gpu_via_predictor
    else:
        t_gpu_gcn = t_gpu_coarse

    elapsed_wall = time.time() - t0_wall

    if _NNI_ACTIVE and nni is not None:
        nni.report_final_result(float(result['test']))
        import os, json
        
        base_log_dir = r"C:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log"
        if not os.path.exists(r"C:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL"):
            base_log_dir = 'log'
            
        os.makedirs(base_log_dir, exist_ok=True)
        trial_id = nni.get_trial_id()
        acc      = float(result['test'])

        try:
            log_file = os.path.join(base_log_dir, f"nni_timing_{args.dataset}_{args.noise_type}_{args.noise_rate}.log")
            import time as tm
            with open(log_file, 'a', encoding='utf-8') as f:
                if t_cpu_pcc > 0:
                    f.write(
                        f"Trial: {trial_id} | Wall: {elapsed_wall:.2f}s | "
                        f"LN-PCC(CPU): {t_cpu_pcc:.2f}s | GCN(GPU): {t_gpu_gcn:.2f}s | "
                        f"Acc: {acc:.4f} | Params: {json.dumps(nni_params)}\n"
                    )
                else:
                    f.write(
                        f"Trial: {trial_id} | Wall: {elapsed_wall:.2f}s | "
                        f"GCN(GPU): {t_gpu_gcn:.2f}s | "
                        f"Acc: {acc:.4f} | Params: {json.dumps(nni_params)}\n"
                    )

            csv_file  = os.path.join(base_log_dir, 'nni_trials.csv')
            csv_exists = os.path.exists(csv_file)
            flat_p     = nni_params if isinstance(nni_params, dict) else {}
            row = {
                'timestamp':  tm.strftime('%Y-%m-%d %H:%M:%S'),
                'trial_id':   trial_id,
                'dataset':    args.dataset,
                'noise_type': args.noise_type,
                'noise_rate': args.noise_rate,
                'acc':        round(acc, 6),
                'wall_s':     round(elapsed_wall, 2),
                'pcc_cpu_s':  round(t_cpu_pcc, 2) if t_cpu_pcc else '',
                'gcn_gpu_s':  round(t_gpu_gcn, 2) if t_gpu_gcn else '',
            }
            # Add hyperparameters as extra columns
            for pk, pv in flat_p.items():
                row[f'p_{pk}'] = pv

            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction='ignore')
                if not csv_exists:
                    writer.writeheader()
                writer.writerow(row)

            print(
                f"[Trial {trial_id}]  acc={acc:.4f}  "
                f"wall={elapsed_wall:.1f}s  pcc={t_cpu_pcc:.1f}s  gcn={t_gpu_gcn:.1f}s",
                flush=True
            )
        except Exception as e:
            print(f"[Trial {trial_id}] Warning: Error during final logging: {e}", flush=True)

    if _use_cuda:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
