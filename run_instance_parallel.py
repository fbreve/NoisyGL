"""
run_instance_parallel.py
========================
Parallel launcher for the Instance Noise Experiment (HPO + Benchmark).
Cloned logic from run_lnpcc_parallel.py (LPT scheduling, log tailing, parallel workers).

Methods: lnpcc, gcn, nrgnn, pignn, cp
Noise Type: instance
"""

import argparse
import os
import sys
import io

# Windows Encoding Fix
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except: pass

# Windows Stability Fixes (Must be set before importing torch/numpy)
os.environ['KMP_STACKSIZE'] = '128M'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import csv
import glob
import time
import threading
import subprocess
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_METHODS = ['lnpcc', 'gcn', 'nrgnn', 'pignn', 'cp']
ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'amazoncom', 'amazonpho', 'dblp',
    'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'
]
NOISE_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

# Execution time weights for Longest Processing Time (LPT) scheduling
DATASET_WEIGHTS = {
    'amazon-ratings': 100,
    'roman-empire': 90,
    'amazoncom': 80,
    'amazonpho': 70,
    'flickr': 60,
    'blogcatalog': 50,
    'dblp': 40,
    'pubmed': 30,
    'citeseer': 20,
    'cora': 10,
}

def _worker_init(priority=None):
    """Initializer for ProcessPoolExecutor workers."""
    if priority and priority != 'normal':
        import psutil
        try:
            p = psutil.Process(os.getpid())
            if priority == 'below_normal':
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif priority == 'idle':
                p.nice(psutil.IDLE_PRIORITY_CLASS)
        except Exception:
            pass

def run_worker(job, gpu_id, priority, log_dir, max_trials, storage):
    """
    Runs hyperparam_opt_instance.py for one (method × dataset × rate) job.
    """
    method = job['method']
    dataset = job['dataset']
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Windows Stability: Increase OpenMP stack size to prevent 0xC0000409
    os.environ['KMP_STACKSIZE'] = '128M'
    # Prevent CUDA fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    rate = job['noise_rate']
    
    import logging
    # ── CRITICAL: Reset all logging handlers inherited from parent process ──
    # When ProcessPoolExecutor spawns a worker on Windows, Python's logging
    # module inherits the parent's handlers. Optuna uses logging internally, so any
    # log call causes: ValueError: I/O operation on closed file → KeyboardInterrupt.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try: handler.close()
        except: pass
        root_logger.removeHandler(handler)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lgr = logging.getLogger(name)
        for handler in lgr.handlers[:]:
            try: handler.close()
            except: pass
            lgr.removeHandler(handler)

    scenario_key = f"{method}_{dataset}_{rate}"
    log_file = os.path.join(log_dir, f"worker_{scenario_key}.log")
    
    python_exe = r'C:\Users\fbrev\anaconda3\envs\noisygl\python.exe'
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    cmd = [
        python_exe, 'hyperparam_opt_instance.py',
        '--method', method,
        '--dataset', dataset,
        '--noise_type', 'instance',
        '--noise_rate', str(rate),
        '--device', f'cuda:{gpu_id}',
        '--max_trial_number', str(max_trials),
        '--priority', priority,
        '--storage', storage
    ]
    
    rc = 1
    with open(log_file, 'w', encoding='utf-8', buffering=1) as f:
        f.write(f"[worker] START {scenario_key} gpu={gpu_id}\n")
        f.flush()
        
        try:
            # Launch without global locks
            p = subprocess.Popen(cmd, stdout=f, stderr=f, text=True)
            p.wait()
            rc = p.returncode
        except Exception as e:
            f.write(f"[worker ERROR] {str(e)}\n")
            rc = 1
            
    return job, rc

def _tail_worker_log(log_path, jid, stop_event):
    """Prints important lines from worker log to terminal in real-time."""
    KEYWORDS = ('Trial ', '[HPO]', '[Retest]', '[Optuna]', 'FAILED', 'ERROR', 'Success', 'RESULT', 'Phase', '[cuda:', 'Waiting', 'acquired', '[CPU]', '[LNPCC-CPU]', 'FINAL')
    while not stop_event.is_set() and not os.path.exists(log_path):
        time.sleep(0.5)
    
    if not os.path.exists(log_path): return
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            while not stop_event.is_set():
                line = f.readline()
                if line:
                    line = line.strip()
                    if any(kw in line for kw in KEYWORDS):
                        print(f"  [{jid[:25]:25}] {line}", flush=True)
                else:
                    time.sleep(0.5)
    except: pass

def main():
    parser = argparse.ArgumentParser(description='Parallel launcher for Instance Noise Experiments')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--workers_per_gpu', type=int, default=4,
                        help='Concurrent workers per GPU. 4 allows CPU overlap while GPU is busy (default: 4)')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS)
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS)
    parser.add_argument('--noise_rates', nargs='+', type=float, default=NOISE_RATES)
    parser.add_argument('--priority', type=str, default='below_normal', choices=['normal', 'below_normal', 'idle'])
    parser.add_argument('--max_trials', type=int, default=50)
    parser.add_argument('--resume', action='store_true', help='Skip already optimized scenarios')
    parser.add_argument('--storage', type=str, default='sqlite:///log/optuna_instance.db',
                        help='Optuna storage URL (passed to workers)')
    parser.add_argument('--db_path', type=str, default='log/hpo_db_instance.json',
                        help='Path to the results JSON file')
    parser.add_argument('--output_csv', type=str, default='log/instance_final_results.csv',
                        help='Path to the final output CSV')
    args = parser.parse_args()
    
    # Total workers = workers_per_gpu × number of GPUs
    max_workers = args.workers_per_gpu * len(args.gpus)

    log_dir = './log/instance_parallel_logs'
    done_dir = './log/instance_done'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    t_start = time.time()

    # Build job list
    jobs = []
    for method in args.methods:
        for ds in args.datasets:
            for rate in args.noise_rates:
                scenario_key = f"{method}_{ds}_instance_{rate}"
                if args.resume and os.path.exists(os.path.join(done_dir, f"{scenario_key}.done")):
                    continue
                jobs.append({
                    'method': method,
                    'dataset': ds,
                    'noise_rate': rate
                })

    # LPT Scheduling: Sort by dataset weight descending
    jobs.sort(key=lambda j: DATASET_WEIGHTS.get(j['dataset'], 0), reverse=True)

    print(f"Starting Parallel Instance Noise HPO/Benchmark for {len(jobs)} scenarios...")
    print(f"GPUs: {args.gpus} | Workers/GPU: {args.workers_per_gpu} | Total Workers: {max_workers} | Trials: {args.max_trials}")

    gpu_cycle = [args.gpus[i % len(args.gpus)] for i in range(max_workers)]

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init, initargs=(args.priority,)) as pool:
        futures = {}
        tail_threads = []

        for i, job in enumerate(jobs):
            gpu_id = gpu_cycle[i % len(gpu_cycle)]
            scenario_key = f"{job['method']}_{job['dataset']}_{job['noise_rate']}"
            log_path = os.path.join(log_dir, f"worker_{scenario_key}.log")
            
            # Clean stale log
            if os.path.exists(log_path):
                try: os.remove(log_path)
                except: pass

            print(f" [Launcher] Dispatching worker for {scenario_key} on cuda:{gpu_id}...", flush=True)
            fut = pool.submit(run_worker, job, gpu_id, args.priority, log_dir, args.max_trials, args.storage)
            futures[fut] = scenario_key

            # Start tail thread
            stop_ev = threading.Event()
            t = threading.Thread(target=_tail_worker_log, args=(log_path, scenario_key, stop_ev), daemon=True)
            t.start()
            tail_threads.append((t, stop_ev))
            
            # Stagger startup slightly to prevent Windows DLL loading collisions
            time.sleep(0.5) 

        # Wait for all workers to complete
        done_cnt, fail_cnt = 0, 0
        for future in as_completed(futures):
            job, rc = future.result()
            scenario_key = f"{job['method']}_{job['dataset']}_instance_{job['noise_rate']}"
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            if rc == 0: done_cnt += 1
            else: fail_cnt += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] COMPLETED: {scenario_key} | {status}", flush=True)

        # Stop all tail threads
        for _, ev in tail_threads: ev.set()
        print(f"\n[FINISH] Done: {done_cnt}, Failed: {fail_cnt}", flush=True)

    # ── Generate Final CSV (after all workers are done) ──
    print(f"\n[Launcher] Generating final consolidated CSV...", flush=True)
    generate_final_csv_from_db('./log/hpo_db_instance.json', './log/instance_final_results.csv')

    elapsed = time.time() - t_start
    print(f"\n[OK] All scenarios completed in {elapsed/3600:.1f} hours.")
    print(f"Final results saved to: ./log/instance_final_results.csv")

def generate_final_csv_from_db(db_path, output_csv):
    import json, csv, numpy as np
    if not os.path.exists(db_path):
        print(f" [Error] DB not found for CSV generation: {db_path}")
        return
    
    with open(db_path, 'r') as f:
        db = json.load(f)
    
    fieldnames = ['method', 'dataset', 'noise_rate', 'acc_mean', 'acc_std', 'wall_s_avg', 'cpu_s_avg', 'gpu_s_avg', 'wait_s_avg']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort keys to have a organized table
        for key in sorted(db.keys()):
            entry = db[key]
            # Key format: method_dataset_noise-type_noise-rate
            parts = key.split('_')
            # Extract basic info
            method = parts[0]
            dataset = parts[1]
            rate = parts[-1]
            
            # Calculate averages of times from all_final_results
            finals = entry.get('all_final_results', [])
            if finals:
                walls = [r.get('wall_s', 0) for r in finals]
                cpus = [r.get('cpu_s', 0) for r in finals]
                gpus = [r.get('gpu_s', 0) for r in finals]
                waits = [r.get('wait_s', 0) for r in finals]
                
                writer.writerow({
                    'method': method,
                    'dataset': dataset,
                    'noise_rate': rate,
                    'acc_mean': f"{entry['retest_avg']:.4f}",
                    'acc_std': f"{entry['retest_std']:.4f}",
                    'wall_s_avg': f"{np.mean(walls):.2f}",
                    'cpu_s_avg': f"{np.mean(cpus):.2f}",
                    'gpu_s_avg': f"{np.mean(gpus):.2f}",
                    'wait_s_avg': f"{np.mean(waits):.2f}"
                })
    print(f" [OK] Consolidated CSV created: {output_csv}")

if __name__ == '__main__':
    main()
