"""
run_lnpcc_parallel.py
Parallel launcher for the GCN + LN-PCC (Unified) NoisyGL benchmark.

Architecture
────────────
Two models: GCN (baseline) and LNPCC (unified NNI-optimized).

Job granularity:
  # Phase 1 only: HPO optimization (capped managers, 200 trials each)
  python run_lnpcc_parallel.py --all_datasets --hpo_only --optimize_trials 200

  # Phase 2: benchmark (skip HPO, use pre-tuned YAMLs, reuse GCN)
  python run_lnpcc_parallel.py --all_datasets --all_noise \\
      --gpus 0 1 --skip_hpo --skip_gcn --max_workers 16 --resume
"""

import argparse
import os
import sys
import csv
import glob
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_VARIANTS = ['lnpcc', 'gcn']
DEFAULT_VARIANTS = ['lnpcc']

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'amazoncom', 'amazonpho', 'dblp',
    'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire',
]

NOISE_TYPE_ORDER = ['clean', 'uniform', 'pair', 'random']

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
        import os
        try:
            p = psutil.Process(os.getpid())
            if priority == 'below_normal':
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif priority == 'idle':
                p.nice(psutil.IDLE_PRIORITY_CLASS)
        except Exception:
            pass
    # Shared semaphores are no longer used; workers use file-based GPULock.
    pass


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


class TeeLogger(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return getattr(self.files[0], 'encoding', 'utf-8')
    @property
    def errors(self):
        return getattr(self.files[0], 'errors', 'strict')


def run_worker(job_key, gpu_id, args_passthrough, partial_dir, log_dir):
    """
    Runs total_exp_lnpcc.py for one (variant × dataset × noise_type × noise_rate) job.
    job_key: dict with keys variant, dataset, noise_type, noise_rate
    Returns (job_key, returncode, partial_csv_path).
    """
    import sys as _sys
    import logging

    # ── CRITICAL: Reset all logging handlers inherited from parent process ──
    # When ProcessPoolExecutor spawns a worker on Windows, Python's logging
    # module inherits the parent's handlers (e.g. StreamHandler pointing to
    # a closed TeeLogger or file). Optuna uses logging internally, so any
    # log call causes: ValueError: I/O operation on closed file → KeyboardInterrupt.
    # We must nuke all inherited handlers and replace with a clean file handler.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    # Also reset all other named loggers (e.g. optuna's own logger)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lgr = logging.getLogger(name)
        for handler in lgr.handlers[:]:
            try:
                handler.close()
            except Exception:
                pass
            lgr.removeHandler(handler)

    variant    = job_key['variant']
    dataset    = job_key['dataset']
    noise_type = job_key['noise_type']
    noise_rate = job_key.get('noise_rate')

    # Build a compact job id for partial file naming
    if variant == 'lnpcc' and args_passthrough and '--hpo_only' in args_passthrough:
        # For HPO jobs, we want the id to reflect the specific scenario being optimized
        job_id = f'hpo_{dataset}_{noise_type}_{noise_rate}'
    elif noise_rate is not None:
        job_id = f'{variant}_{dataset}_{noise_type}_{noise_rate}'
    else:
        job_id = f'{variant}_{dataset}_{noise_type}'

    partial_csv = os.path.join(partial_dir, f'partial_{job_id}.csv')
    log_file    = os.path.join(log_dir, f'worker_{job_id}.log')

    # ── Route all worker output (stdout, stderr, logging) to the log file ──
    # We write directly to the log_file, and the parent's tail thread prints to the terminal.
    import subprocess
    import sys as _sys
    
    rc = 1
    with open(log_file, 'w', encoding='utf-8', buffering=1) as _log_fh:
        _log_fh.write(f'[worker] START  {job_id}  gpu={gpu_id}\n')
        _log_fh.flush()

        t0 = time.time()

        # Build args: this worker runs exactly one (dataset, noise_type, noise_rate)
        cmd = [
            _sys.executable, 'total_exp_lnpcc.py',
            '--variants', variant,
            '--datasets', dataset,
            '--noise_type', noise_type,
            '--device', f'cuda:{gpu_id}',
            '--partial_csv', partial_csv,
        ]
        if noise_rate is not None:
            cmd.extend(['--noise_rate', str(noise_rate)])
        cmd.extend(args_passthrough)

        env = os.environ.copy()
        env['LNPCC_N_JOBS'] = '1'

        max_scenario_retries = 3
        for attempt_idx in range(max_scenario_retries):
            try:
                proc = subprocess.run(cmd, env=env, stdout=_log_fh, stderr=subprocess.STDOUT)
                rc = proc.returncode
                if rc == 0:
                    break
                _log_fh.write(f'\n[worker NATIVE CRASH] Process died (rc={rc}) on attempt {attempt_idx+1}/{max_scenario_retries}. Retrying the full scenario...\n')
                _log_fh.flush()
                time.sleep(2)
            except Exception as exc:
                _log_fh.write(f'\n[worker INIT ERROR] {exc}\n')
                rc = 1
                break

        elapsed = time.time() - t0
        status = 'OK' if rc == 0 else 'FAIL'
        _log_fh.write(f'\n[worker] {status} DONE  {job_id}  gpu={gpu_id}  {elapsed/60:.1f} min  rc={rc}\n')
        _log_fh.flush()

    return job_key, rc, partial_csv


# ── Merge partial CSVs ────────────────────────────────────────────────────────

def merge_partials(partial_dir, output_csv):
    """
    Merges all partial_*.csv files into a single wide CSV.
    Each partial has columns: dataset, noise_rate, <method>_mean, <method>_std.
    """
    partial_files = sorted(glob.glob(os.path.join(partial_dir, 'partial_*.csv')))
    if not partial_files:
        print('[launcher] No partial CSVs to merge.', flush=True)
        return

    merged = {}
    all_value_cols = []

    for pf in partial_files:
        with open(pf, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['dataset'], row['noise_rate'])
                if key not in merged:
                    merged[key] = {'dataset': row['dataset'],
                                   'noise_rate': row['noise_rate']}
                for col, val in row.items():
                    if col not in ('dataset', 'noise_rate'):
                        merged[key][col] = val
                        if col not in all_value_cols:
                            all_value_cols.append(col)

    fieldnames = ['dataset', 'noise_rate'] + sorted(all_value_cols)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for key in sorted(merged.keys()):
            writer.writerow(merged[key])

    print(f'[launcher] Merged {len(partial_files)} partial CSVs -> {output_csv}', flush=True)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Parallel launcher for GCN + Unified LN-PCC NoisyGL benchmark'
    )
    # Launcher-specific
    p.add_argument('--gpus', type=int, nargs='+', default=[0])
    p.add_argument('--max_workers', type=int, default=16,
                   help='Max concurrent workers (default: 16)')
    p.add_argument('--priority', type=str, default='below_normal',
                   choices=['normal', 'below_normal', 'idle'],
                   help='Process priority for launcher and workers (default: below_normal)')
    p.add_argument('--resume', action='store_true',
                   help='Skip jobs whose partial CSV already exists and is complete')
    p.add_argument('--output', type=str, default=None,
                   help='Final merged CSV path (default: ./log/lnpcc_results_<ts>.csv)')
    p.add_argument('--skip_gcn', action='store_true',
                   help='Reuse existing log/partial/partial_gcn_*.csv instead of re-running GCN. '
                        'If no GCN partial exists, GCN will run normally.')

    # Passed through to total_exp_lnpcc.py workers
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--seed', type=int, default=3000)
    p.add_argument('--variants', type=str, nargs='+', default=DEFAULT_VARIANTS,
                   choices=ALL_VARIANTS)
    p.add_argument('--datasets', nargs='+', default=['cora', 'citeseer', 'pubmed'],
                   choices=ALL_DATASETS)
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--noise_type', nargs='+',
                   default=['clean', 'uniform', 'pair', 'random'],
                   choices=['clean', 'uniform', 'pair', 'random', 'instance'])
    p.add_argument('--all_noise', action='store_true',
                   help='Run clean, uniform, pair, and random noise types.')
    p.add_argument('--noise_rate', nargs='+', type=float,
                   default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument('--optimize', action='store_true',
                   help='Run Optuna HPO for lnpcc before benchmark')
    p.add_argument('--hpo_only', action='store_true',
                   help='PHASE 1: Only run HPO optimization, skip benchmark')
    p.add_argument('--skip_hpo', action='store_true',
                   help='PHASE 2: Skip HPO optimization, only run benchmark')
    p.add_argument('--optimize_trials', type=int, default=200)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--retest_runs', type=int, default=10)
    p.add_argument('--max_k_same', type=int, default=None,
                   help='Maximum k for knn_mode="s" HPO search (None=auto by dataset). '
                        'Use 15 for large datasets (amazon-ratings, roman-empire) to prevent '
                        'subprocess timeouts and 0xC0000409 crashes.')
    p.add_argument('--debug', action='store_true')

    return p.parse_args()


def build_passthrough(args):
    """Build the list of CLI args to forward to each worker (excluding per-job args)."""
    parts = ['--runs', str(args.runs), '--seed', str(args.seed)]
    if args.hpo_only:
        # HPO phase: always inject --optimize so workers actually run Optuna
        parts += ['--optimize', '--hpo_only',
                  '--optimize_trials', str(args.optimize_trials)]
    elif args.skip_hpo:
        parts.append('--skip_hpo')
    elif args.optimize:
        parts += ['--optimize',
                  '--optimize_trials', str(args.optimize_trials)]
    parts += ['--top_k', str(args.top_k), '--retest_runs', str(args.retest_runs)]
    if args.max_k_same is not None:
        parts += ['--max_k_same', str(args.max_k_same)]
    if args.debug:
        parts.append('--debug')
    return parts


# ── Job builder ───────────────────────────────────────────────────────────────

def build_jobs(args, datasets, noise_types, partial_dir):
    """
    Build the list of (job_key, partial_csv) tuples.

    Phase 1 (--nni_only): one job per dataset — NNI runs once per dataset.
      Each worker receives: --variants lnpcc --datasets <ds> --optimize --nni_only
    Phase 2 (benchmark): one job per (variant × dataset × noise_type × noise_rate).
      GCN jobs may be skipped via --skip_gcn if per-scenario partial CSVs exist.
    """
    jobs = []

    if args.hpo_only:
        # ── HPO phase: one job per scenario (dataset × noise_type × noise_rate) ──
        for dataset in datasets:
            for noise_type in noise_types:
                rates = [0.0] if noise_type == 'clean' else args.noise_rate
                for noise_rate in rates:
                    job_id      = f'hpo_{dataset}_{noise_type}_{noise_rate}'
                    partial_csv = os.path.join(partial_dir, f'partial_{job_id}.csv')
                    
                    # Skip if HPO done-marker exists for this specific scenario
                    done_marker = os.path.join('./log/nni_done', f'lnpcc_{dataset}_{noise_type}_{noise_rate}.done')
                    if args.resume and os.path.exists(done_marker):
                        print(f'[launcher] SKIP_HPO {dataset} {noise_type} {noise_rate} (done marker exists)', flush=True)
                        continue
                        
                    job_key = {'variant': 'lnpcc', 'dataset': dataset,
                               'noise_type': noise_type, 'noise_rate': noise_rate}
                    jobs.append((job_key, partial_csv))
        return jobs

    # ── Benchmark phase: one job per (variant × dataset × noise_scenario) ──
    for variant in args.variants:
        for dataset in datasets:
            for noise_type in noise_types:
                if noise_type == 'clean':
                    noise_rates = [None]  # clean has no rate (always 0.0)
                else:
                    noise_rates = args.noise_rate

                for noise_rate in noise_rates:
                    if noise_rate is not None:
                        job_id = f'{variant}_{dataset}_{noise_type}_{noise_rate}'
                    else:
                        job_id = f'{variant}_{dataset}_{noise_type}'

                    partial_csv = os.path.join(partial_dir, f'partial_{job_id}.csv')

                    # Skip GCN if requested and per-scenario partial already exists
                    if variant == 'gcn' and args.skip_gcn:
                        if os.path.exists(partial_csv):
                            print(f'[launcher] SKIP_GCN {job_id} (partial exists)', flush=True)
                            continue
                        else:
                            print(f'[launcher] SKIP_GCN requested but no partial for {job_id} — will run GCN', flush=True)

                    # Skip if --resume and partial is complete
                    if args.resume and os.path.exists(partial_csv):
                        try:
                            with open(partial_csv, 'r', encoding='utf-8') as _f:
                                rows = list(csv.DictReader(_f))
                            if len(rows) > 0:
                                mean_col = f'{variant}_mean'
                                if mean_col in rows[0] and rows[0].get(mean_col, '') != '':
                                    print(f'[launcher] SKIP   {job_id} (partial CSV complete)', flush=True)
                                    continue
                        except Exception:
                            pass

                    job_key = {
                        'variant': variant,
                        'dataset': dataset,
                        'noise_type': noise_type,
                        'noise_rate': noise_rate,
                    }
                    jobs.append((job_key, partial_csv))

    return jobs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_priority(args.priority)

    datasets    = ALL_DATASETS if args.all_datasets else args.datasets
    noise_types = ['clean', 'uniform', 'pair', 'random'] \
                  if args.all_noise else args.noise_type

    gpus        = args.gpus
    max_workers = args.max_workers

    # ── NNI Phase Optimization ───────────────────────────────────────────────
    # If running NNI phase, we want to prevent 10 managers starting together 
    # with 1 trial each, as it causes tail-end stragglers. Instead, we cap
    # managers to `len(gpus)` and boost NNI trial concurrency per manager.
    if args.hpo_only:
        # EXHAUSTIVE HPO: We want to run many scenarios in parallel,
        # but each scenario is internally sequential.
        # So we use the full max_workers for the launcher.
        print(f'[launcher] Exhaustive HPO: Running {args.max_workers} scenarios in parallel '
              f'with sequential trials.', flush=True)

    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir     = './log'
    partial_dir = os.path.join(log_dir, 'partial')
    output_csv  = args.output or os.path.join(log_dir, f'lnpcc_results_{ts}.csv')
    os.makedirs(partial_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Clean up any stale GPU lock directories from previous runs ──
    # These are created by GPULock and must be deleted if a run crashed.
    # A lock is stale if it exists before this launcher starts (any age).
    import shutil
    for _gpu_id in range(8):  # covers GPU 0..7
        _lock_path = os.path.join(log_dir, f'gpu_{_gpu_id}.lock')
        if os.path.isdir(_lock_path):
            try:
                shutil.rmtree(_lock_path)
                print(f'[launcher] Removed stale GPU lock: {_lock_path}', flush=True)
            except Exception as _e:
                print(f'[launcher] WARNING: could not remove stale lock {_lock_path}: {_e}', flush=True)
    
    # Note: spawn_lock is no longer used for global initialization serialization.


    # ── Central Execution Log ──────────────────────────────────────────
    launcher_log = os.path.join(log_dir, f'launcher_{ts}.log')
    # Parent logging: just keep it in one file, don't wrap sys.stdout with custom objects.
    # This prevents serialization failures when spawning children on Windows.
    print(f'[launcher] Logging to {launcher_log}', flush=True)

    passthrough = build_passthrough(args)

    print('=' * 70)
    print('GCN + LN-PCC (Unified) -- Parallel Benchmark Launcher')
    print(f'Start      : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Datasets   : {datasets}')
    print(f'Noise types: {noise_types}')
    print(f'GPUs       : {gpus}')
    print(f'Max workers: {max_workers}')
    print(f'Skip GCN   : {args.skip_gcn}')
    print(f'Passthrough: {" ".join(passthrough)}')
    print(f'Output CSV : {output_csv}')
    print('=' * 70, flush=True)

    # ── Build jobs ─────────────────────────────────────────────────────
    all_jobs = build_jobs(args, datasets, noise_types, partial_dir)

    # LPT Scheduling: Sort jobs by dataset weight descending so longest run first.
    all_jobs.sort(key=lambda j: DATASET_WEIGHTS.get(j[0]['dataset'], 0), reverse=True)

    if not all_jobs:
        print('[launcher] Nothing to run (all partials exist or skipped). Merging...', flush=True)
        merge_partials(partial_dir, output_csv)
        return

    print(f'[launcher] {len(all_jobs)} jobs queued', flush=True)

    # ── Semaphores ─────────────────────────────────────────────────────
    gpu_assignment = [gpus[i % len(gpus)] for i in range(max_workers)]

    print(f'[launcher] GPU cycle: {gpu_assignment[:min(len(all_jobs),8)]}...', flush=True)

    # ── Run workers ────────────────────────────────────────────────────
    failed  = []
    t_start = time.time()
    import threading

    def _tail_worker_log(log_path, jid, stop_event):
        """
        Background thread: tail a worker log file and print lines to the terminal.
        This is necessary because ProcessPoolExecutor child stdout is discarded on Windows.
        Only prints lines containing keywords to avoid noise.
        """
        KEYWORDS = ('[Trial', '[worker]', '[HPO]', '[Retest]', '[Optuna]',
                    '[Phase', '[LN-PCC]', '[CHECKPOINT]', 'FAILED', 'ERROR',
                    'native crash', 'timeout', 'Started', 'Done',
                    '| run ', '-> mean=')
        import builtins
        # Wait for the log file to be created (worker opens it in 'w' before writing)
        # We must wait indefinitely until stop_event is set, because jobs may sit in the executor queue for hours!
        while not stop_event.is_set() and not os.path.exists(log_path):
            time.sleep(0.5)
        
        if not os.path.exists(log_path):
            return
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                # Read from position 0: the old log was deleted before job submission,
                # so this is always a fresh file written by the current worker.
                while not stop_event.is_set():
                    line = f.readline()
                    if line:
                        line = line.rstrip()
                        if any(kw in line for kw in KEYWORDS):
                            builtins.print(f'  [{jid[:30]:30}] {line}', flush=True)
                    else:
                        try:
                            current_pos = f.tell()
                            # If the worker just started and truncated the stale file, 
                            # our reader is stranded past the new EOF. Reset to 0!
                            if current_pos > os.path.getsize(log_path):
                                f.seek(0)
                            else:
                                f.seek(current_pos)  # Clear EOF flag for Windows
                        except Exception:
                            # File might be temporarily locked or unavailable
                            f.seek(f.tell())
                        time.sleep(0.5)
        except Exception:
            pass

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(args.priority,),
    ) as pool:
        futures = {}
        tail_threads = []

        for i, (job_key, _partial) in enumerate(all_jobs):
            assigned_gpu = gpu_assignment[i % len(gpu_assignment)]
            
            v_var = job_key["variant"]
            v_ds = job_key["dataset"]
            v_nt = job_key["noise_type"]
            v_nr = job_key.get("noise_rate")
            
            if v_var == 'lnpcc' and passthrough and '--hpo_only' in passthrough:
                jid = f'hpo_{v_ds}_{v_nt}_{v_nr}'
            elif v_nr is not None:
                jid = f'{v_var}_{v_ds}_{v_nt}_{v_nr}'
            else:
                jid = f'{v_var}_{v_ds}_{v_nt}'
                
            log_file = os.path.join(log_dir, f'worker_{jid}.log')

            # Delete any stale log file from a previous run so the tail thread
            # always opens a fresh file and reads from position 0, never from
            # a position beyond the EOF of the truncated file.
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
            except Exception:
                pass

            fut = pool.submit(
                run_worker, job_key, assigned_gpu,
                passthrough, partial_dir, log_dir
            )
            futures[fut] = job_key

            # Start a tail thread for this worker's log
            stop_ev = threading.Event()
            t = threading.Thread(
                target=_tail_worker_log,
                args=(log_file, jid, stop_ev),
                daemon=True
            )
            t.start()
            tail_threads.append((t, stop_ev))

        for future in as_completed(futures):
            job_key, rc, partial_csv = future.result()
            
            v_var = job_key["variant"]
            v_ds = job_key["dataset"]
            v_nt = job_key["noise_type"]
            v_nr = job_key.get("noise_rate")
            
            if v_var == 'lnpcc' and passthrough and '--hpo_only' in passthrough:
                jid = f'hpo_{v_ds}_{v_nt}_{v_nr}'
            elif v_nr is not None:
                jid = f'{v_var}_{v_ds}_{v_nt}_{v_nr}'
            else:
                jid = f'{v_var}_{v_ds}_{v_nt}'
                
            status = 'OK' if rc == 0 else 'FAIL'
            print(f'[launcher] {status}    {jid:50} rc={rc}', flush=True)
            if rc != 0:
                failed.append(jid)

        # Stop all tail threads immediately without sequential blocking
        for t, stop_ev in tail_threads:
            stop_ev.set()

    elapsed = time.time() - t_start
    print(f'\n[launcher] All workers done in {elapsed/60:.1f} min', flush=True)
    if failed:
        print(f'[launcher] WARNING: {len(failed)} job(s) failed: {failed}', flush=True)

    merge_partials(partial_dir, output_csv)

    print(f'\n{"="*70}')
    print(f'[OK] Final CSV: {output_csv}')
    print(f'   Worker logs: {log_dir}/worker_<jobid>.log')
    if failed:
        print(f'   Failed jobs: {failed}')
        print(f'   Re-run with --resume to retry only failed jobs.')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
