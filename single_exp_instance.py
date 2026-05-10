import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings("ignore", message=".*Converting sparse tensor to CSR format.*")
import time
import threading
import json
import csv
import traceback
import math
import io
from filelock import FileLock

# Windows Stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Windows Encoding Fix
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except: pass

from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed
from utils.labelnoise import label_process
from predictor.NRGNN_Predictor_Instance import nrgnn_Predictor
from predictor.CP_Predictor import cp_Predictor
from predictor.GCN_Predictor import gcn_Predictor
from predictor.PIGNN_Predictor_Instance import pignn_Predictor
# LNPCC uses the original predictor which manages its own GPU lock internally
from predictor.LNPCC_Predictor import lnpcc_Predictor

def set_priority(level):
    if level == 'normal': return
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if level == 'below_normal': p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif level == 'idle': p.nice(psutil.IDLE_PRIORITY_CLASS)
    except: pass

def flatten_nni_params(params):
    flat = {}
    if not params: return flat
    for k, v in params.items():
        if isinstance(v, dict) and '_name' in v:
            flat[k] = v['_name']
            for sub_k, sub_v in v.items():
                if sub_k != '_name': flat[sub_k] = sub_v
        else: flat[k] = v
    return flat

def _apply_flat_params(model_conf, flat_params):
    for item, val in flat_params.items():
        if item in ['lr', 'weight_decay']: model_conf.training[item] = val
        else: model_conf.model[item] = val

def _safe_float(val, default=0.0):
    """Return val as float, or default if None/NaN."""
    if val is None: return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except: return default

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dblp')
    parser.add_argument('--method', type=str, default='gcn')
    parser.add_argument('--noise_type', type=str, default='instance')
    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--params_json', type=str, default=None)
    parser.add_argument('--priority', type=str, default='normal')
    parser.add_argument('--trial_id', type=str, default='x')
    args = parser.parse_args()

    # Force this process to use the assigned GPU as its default 'cuda' device.
    # This fixes issues where external modules use .cuda() without arguments.
    if 'cuda' in args.device:
        try:
            gpu_id = int(args.device.split(':')[-1])
            torch.cuda.set_device(gpu_id)
        except: pass

    set_priority(args.priority)
    t0_wall = time.time()
    wait_duration = 0.0
    result_container = {}

    try:
        # ── 1. CPU Phase (Fully Parallel for all methods) ──
        print(f" [CPU] Loading {args.dataset}...", flush=True)
        data_conf = load_conf('./config/_dataset/' + args.dataset + '.yaml')
        setup_seed(args.seed)
        
        data = Dataset(args.dataset, path='./data/', device='cpu', 
                       feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                       train_size=data_conf.split['train_size'], val_size=data_conf.split['val_size'],
                       test_size=data_conf.split['test_size'],
                       train_percent=data_conf.split['train_percent'], val_percent=data_conf.split['val_percent'],
                       test_percent=data_conf.split['test_percent'],
                       train_examples_per_class=data_conf.split['train_examples_per_class'],
                       val_examples_per_class=data_conf.split['val_examples_per_class'],
                       test_examples_per_class=data_conf.split['test_examples_per_class'],
                       add_self_loop=data_conf.modify['add_self_loop'],
                       from_npz=data_conf.modify['from_npz_largest_component'],
                       split_type=data_conf.split['split_type'])

        t0_cpu = time.time()
        data.noisy_label, modified_mask = label_process(
            labels=data.labels, features=data.feats,
            n_classes=data.n_classes, noise_type=args.noise_type,
            noise_rate=args.noise_rate, random_seed=args.seed, debug=True)

        # ── 2. Model Config ──
        model_conf = load_conf(None, args.method, data.name)
        if args.params_json:
            p = json.loads(args.params_json)
            flat = flatten_nni_params(p)
            _apply_flat_params(model_conf, flat)

        model_conf.model['n_feat'] = data.dim_feats
        model_conf.model['n_classes'] = data.n_classes
        model_conf.training['debug'] = False  # epoch logs suppressed; set True to debug a single run

        t_cpu_final = time.time() - t0_cpu
        t_gpu_final = 0.0

        # ── 3. LNPCC: manages its own GPU lock internally ──
        if args.method == 'lnpcc':
            predictor = lnpcc_Predictor(model_conf, data, args.device)
            predictor.modified_mask = modified_mask

            threading.stack_size(32 * 1024 * 1024)
            def run_lnpcc():
                try:
                    res = predictor.train()
                    result_container['result'] = res
                    result_container['p_state'] = {
                        't_cpu': t_cpu_final + getattr(predictor, '_t_pcc', 0.0),
                        't_gpu': getattr(predictor, '_t_gcn', 0.0),
                        't_wait': getattr(predictor, '_t_wait', 0.0),
                        'stats': getattr(predictor, '_pcc_stats', None)
                    }
                except Exception as e:
                    result_container['error'] = f"{str(e)}\n{traceback.format_exc()}"
            t = threading.Thread(target=run_lnpcc)
            t.start()
            t.join()

        # ── 4. All other methods: use external FileLock for GPU serialization ──
        else:
            os.makedirs('log', exist_ok=True)
            lock_file = os.path.join('log', f"gpu_{args.device.replace(':', '_')}.lock")
            
            print(f" [{args.device}] Waiting for GPU slot...", flush=True)
            t_lock_start = time.time()
            with FileLock(lock_file, timeout=7200):
                wait_duration = time.time() - t_lock_start
                print(f" [{args.device}] Slot acquired (waited {wait_duration:.1f}s). Training...", flush=True)

                # Initialize predictor INSIDE the lock. 
                # This ensures features/models are moved to GPU only when we have the slot.
                if args.method == 'gcn': predictor = gcn_Predictor(model_conf, data, args.device)
                elif args.method == 'nrgnn': predictor = nrgnn_Predictor(model_conf, data, args.device)
                elif args.method == 'pignn': predictor = pignn_Predictor(model_conf, data, args.device)
                elif args.method == 'cp': predictor = cp_Predictor(model_conf, data, args.device)
                else: predictor = eval(args.method + '_Predictor')(model_conf, data, args.device)
                
                predictor.modified_mask = modified_mask
                
                threading.stack_size(32 * 1024 * 1024)
                def run_train():
                    try:
                        t0_gpu = time.time()
                        res = predictor.train()
                        t_gpu = time.time() - t0_gpu
                        result_container['result'] = res
                        # Detailed timings from the predictor (including internal accumulation)
                        result_container['p_state'] = {
                            't_cpu': t_cpu_final + getattr(predictor, '_t_cpu', 0.0),
                            't_gpu': getattr(predictor, '_t_gpu', t_gpu),
                            'stats': None
                        }
                    except Exception as e:
                        result_container['error'] = f"{str(e)}\n{traceback.format_exc()}"
                t = threading.Thread(target=run_train)
                t.start()
                t.join()

        # ── 5. Results ──
        if 'error' in result_container:
            print(f" [ERROR] {result_container['error']}", flush=True)
            sys.exit(1)

        result = result_container.get('result', {'train': 0.0, 'valid': 0.0, 'test': 0.0})
        p_state = result_container.get('p_state', {'t_cpu': 0.0, 't_gpu': 0.0, 'stats': None})

        # Sanitize NaNs
        for k in ['train', 'valid', 'test']:
            result[k] = _safe_float(result.get(k))

        elapsed_wall = (time.time() - t0_wall) - wait_duration - _safe_float(p_state.get('t_wait', 0.0))
        t_cpu_log = _safe_float(p_state.get('t_cpu'))
        t_gpu_log = _safe_float(p_state.get('t_gpu'))

        csv_file = './log/instance_trials_all.csv'
        csv_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'method', 'dataset', 'noise_rate', 'acc', 'wall_s', 'cpu_s', 'gpu_s', 'wait_s'])
            if not csv_exists: writer.writeheader()
            writer.writerow({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': args.method, 'dataset': args.dataset, 'noise_rate': args.noise_rate,
                'acc': round(result['test'], 6),
                'wall_s': round(elapsed_wall, 2),
                'cpu_s': round(t_cpu_log, 2),
                'gpu_s': round(t_gpu_log, 2),
                'wait_s': round(wait_duration + _safe_float(p_state.get('t_wait', 0.0)), 2)
            })
        
        full_res = {
            'acc': result,
            'wall_s': round(elapsed_wall, 2),
            'cpu_s': round(t_cpu_log, 2),
            'gpu_s': round(t_gpu_log, 2),
            'wait_s': round(wait_duration + _safe_float(p_state.get('t_wait', 0.0)), 2)
        }
        print(f"FINAL_RESULT: {full_res}", flush=True)
        print(f"  [Time] wall={elapsed_wall:.1f}s | cpu={t_cpu_log:.1f}s | gpu={t_gpu_log:.1f}s", flush=True)

    except Exception as e:
        print(f" [FATAL] {str(e)}\n{traceback.format_exc()}", flush=True)
        sys.exit(1)
    finally:
        try: torch.cuda.empty_cache()
        except: pass
