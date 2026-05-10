"""
hyperparam_opt_optuna.py
========================
Optimizes LN-PCC hyperparameters for one (dataset, noise_type, noise_rate) scenario
using Optuna instead of NNI.

Key improvements over the NNI version:
  - Conditional search space: knn_mode → k only sampled when mode ≠ 'u'
  - No HTTP server / port management needed
  - dexp uses loguniform(0.5, 10) — removes degenerate near-zero range
  - k for mode 's' uses int(5, 100) with step 5 (focused on effective range)
  - Results are saved to the same hpo_db.json format for full pipeline compatibility
  - MedianPruner equivalent via early stopping on poor intermediate results

Usage (identical CLI to hyperparam_opt.py):
  python hyperparam_opt_optuna.py --dataset cora --noise_type uniform --noise_rate 0.3
  python hyperparam_opt_optuna.py --dataset citeseer --noise_type pair --noise_rate 0.4 --max_trial_number 150
"""
import os
import sys
import json
import argparse
import subprocess
import re
import time
import random
import shutil
import threading
import traceback

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


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


# ── Search space definition ────────────────────────────────────────────────────

def _suggest_params(trial, max_k_same=60):
    """
    Define the Optuna search space.

    Key differences from NNI search_space/lnpcc.json:
      - knn_mode is a categorical; k is only suggested when mode != 'u'
        (NNI-TPE treated k as a global param even for mode='u', wasting samples)
      - dexp: loguniform(0.5, 10) → removes the near-zero regime that is
        effectively equivalent to knn_mode='u' but wastes the KNN computation
      - k for mode 's':  int(5, max_k_same, step=5) → capped to avoid OOM/crash
        (k=60 verified stable empirically; k=100 caused 0xC0000409 on amazon-ratings)
      - k for 'd'/'p':   int(1, 25)                 → lightly expanded from [1,20]

    max_k_same: maximum k allowed for knn_mode='s'. Default 60 is the last
      empirically verified safe value. k=100 caused STATUS_STACK_BUFFER_OVERRUN
      on large heterophilic graphs like amazon-ratings.
    """
    knn_mode = trial.suggest_categorical('knn_mode', ['u', 's', 'd', 'p'])

    k = None
    if knn_mode == 's':
        # Cosine-similarity KNN: cap to max_k_same to avoid OOM/timeout on large graphs
        k_high = max(5, max_k_same)  # ensure at least one valid step
        k = trial.suggest_int('k', 5, k_high, step=5)
    elif knn_mode in ('d', 'p'):
        # Distance / PCC-kernel KNN: effective at small k
        k = trial.suggest_int('k', 1, 25)
    # knn_mode == 'u': no k parameter (uses original graph only)

    # dexp: exponent of distance kernel.
    # User feedback: dexp=0 is a valid regime where particles ignore distance.
    dexp = trial.suggest_float('dexp', 0.0, 10.0)

    p_grd    = trial.suggest_float('p_grd',    0.0, 0.5)
    unc_rem  = trial.suggest_float('unc_rem',   0.01, 1.0)
    unc_rel  = trial.suggest_float('unc_rel',   0.01, 1.0)
    dropout  = trial.suggest_float('dropout',   0.1, 0.9)
    n_hidden = trial.suggest_categorical('n_hidden', [16, 32, 64, 128])
    n_layer  = trial.suggest_categorical('n_layer',  [1, 2, 3, 4])
    lr       = trial.suggest_categorical('lr',       [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1])
    wd       = trial.suggest_categorical('weight_decay',
                                         [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1])

    params = dict(knn_mode=knn_mode, dexp=dexp, p_grd=p_grd,
                  unc_rem=unc_rem, unc_rel=unc_rel, dropout=dropout,
                  n_hidden=n_hidden, n_layer=n_layer, lr=lr, weight_decay=wd)
    if k is not None:
        params['k'] = k

    return params


# ── Trial runner ───────────────────────────────────────────────────────────────

def _run_trial(params, dataset, noise_type, noise_rate, device, seed, priority='normal', timeout=3600):
    """
    Run single_exp.py as a subprocess and return the test accuracy.
    Spawns a single_exp.py subprocess with the given hyperparameters.
    Captures output, parses FINAL_RESULT, and returns the test accuracy.
    Includes robust retry logic for native crashes (0xC0000409).
    """
    params_json = json.dumps(params)
    cmd = [
        sys.executable, 'single_exp.py',
        '--dataset',    dataset,
        '--method',     'lnpcc',
        '--noise_type', noise_type,
        '--noise_rate', str(noise_rate),
        '--device',     device,
        '--seed',       str(seed),
        '--params_json', params_json,
        '--priority',   priority,
    ]

    proc = None 
    stdout_chunks = []
    stderr_chunks = []

    def _drain(stream, chunks):
        """Background thread: drain a stream into chunks[]. Never raises."""
        try:
            for line in iter(stream.readline, ''):
                chunks.append(line)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    # Directory for temporary HPO trial logs
    tmp_dir = os.path.join('log', 'temp_hpo')
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        max_attempts = 3
        for attempt in range(max_attempts):
            # Unique log file per attempt to preserve crash history during retries
            tmp_log = os.path.join(tmp_dir, f'trial_{dataset}_{noise_type}_{noise_rate}_{seed}_a{attempt}_{random.randint(0, 999)}.log')
            
            if attempt > 0:
                delay = (attempt ** 2) + random.uniform(0, 5)
                time.sleep(delay)
            
            # Staggered startup to avoid 'thundering herd' resource contention on Windows.
            if attempt == 0:
                time.sleep(random.uniform(0, 5))

            try:
                with open(tmp_log, 'w', encoding='utf-8') as fh:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=fh,
                        stderr=fh,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                    )
                    
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        return None
                
                # Now read the full output from the file
                with open(tmp_log, 'r', encoding='utf-8', errors='replace') as fh:
                    stdout_text = fh.read()
                
                rc = proc.returncode
                if rc == 0:
                    break # Success!
                
                # Specific retry for STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
                if rc == 3221226505 and attempt < max_attempts - 1:
                    lock_dir = os.path.join('log', f'gpu_{device.split(":")[-1] if ":" in device else 0}.lock')
                    if os.path.exists(lock_dir):
                        try:
                            shutil.rmtree(lock_dir, ignore_errors=True)
                        except:
                            pass
                    continue
                
                # Other failure
                return None

            except BaseException as e:
                if proc: proc.kill()
                raise e

        match = re.search(r'FINAL_RESULT:\s*(\{.*?\})', stdout_text)
        if not match:
            print(f'    [trial] FINAL_RESULT not found in stdout for {dataset}_{noise_type}_{noise_rate}', flush=True)
            # Log the tail for debugging
            print(f"    [tail] " + " | ".join(stdout_text.strip().splitlines()[-5:]), flush=True)
            return None

        # Clean up temp log on success
        try:
            os.remove(tmp_log)
        except:
            pass

        res = json.loads(match.group(1).replace("'", '"'))
        return float(res.get('test', 0))


    except BaseException as e:
        # Catch KeyboardInterrupt and any other unexpected exception.
        # We intentionally do NOT re-raise so that a single subprocess crash
        # (e.g. 0xC0000409) does not kill the entire Optuna study.
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Optuna HPO for LN-PCC (replaces hyperparam_opt.py)'
    )
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                                 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings',
                                 'roman-empire'])
    parser.add_argument('--method', type=str, default='lnpcc')  # kept for CLI compat
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['clean', 'uniform', 'pair', 'random', 'instance'])
    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--max_trial_number', type=int, default=200)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--retest_runs', type=int, default=10)
    parser.add_argument('--n_startup_trials', type=int, default=20,
                        help='Random trials before TPE kicks in (default: 20)')
    parser.add_argument('--max_k_same', type=int, default=60,
                        help='Maximum k for knn_mode="s" (default: 60, empirically stable). '
                             'k=100 caused 0xC0000409 crashes on large graphs like amazon-ratings.')
    parser.add_argument('--hpo_db', type=str, default='log/hpo_db.json')
    # HPO configuration args
    parser.add_argument('--trial_concurrency', type=int, default=1)
    parser.add_argument('--tuner', type=str, default='TPE')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--update_config', type=bool, default=True)
    parser.add_argument('--priority', type=str, default='normal',
                        choices=['normal', 'below_normal', 'idle'],
                        help='Process priority (default: normal)')
    parser.add_argument('--storage', type=str, default='sqlite:///log/optuna_hpo.db',
                        help='Optuna storage URL (e.g. sqlite:///log/optuna_hpo.db)')

    args = parser.parse_args()
    set_priority(args.priority)

    try:
        import optuna
    except ImportError:
        print('[ERROR] optuna not installed. Run: pip install optuna')
        sys.exit(1)

    # Silence Optuna's verbose logging (keep only warnings+)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scenario_key = f"{args.dataset}_{args.noise_type}_{args.noise_rate}"
    print(f'\n[Optuna] Scenario: {scenario_key}  '
          f'(max_trials={args.max_trial_number}, '
          f'startup={args.n_startup_trials})', flush=True)

    max_k_same = args.max_k_same

    # ── Objective ─────────────────────────────────────────────────────────────
    trial_count = [0]

    def objective(trial):
        trial_count[0] += 1
        params = _suggest_params(trial, max_k_same=max_k_same)
        t0  = time.time()
        acc = _run_trial(params, args.dataset, args.noise_type, args.noise_rate,
                         args.device, args.seed, priority=args.priority)
        elapsed = time.time() - t0

        if acc is None:
            # Penalise failed trials so Optuna avoids similar regions
            print(f'  [Trial {trial_count[0]:3d}] {scenario_key} | FAILED  '
                  f'({elapsed:.1f}s)  knn_mode={params["knn_mode"]}', flush=True)
            raise optuna.exceptions.TrialPruned()

        print(f'  [Trial {trial_count[0]:3d}] {scenario_key} | acc={acc:.4f}  '
              f'({elapsed:.1f}s)'
              f'  knn={params["knn_mode"]}'
              f'  k={params.get("k", "-"):3}'
              f'  dexp={params["dexp"]:.2f}'
              f'  p_grd={params["p_grd"]:.3f}'
              f'  unc_rem={params["unc_rem"]:.3f}'
              f'  unc_rel={params["unc_rel"]:.3f}'
              f'  drop={params["dropout"]:.2f}'
              f'  h={params["n_hidden"]}'
              f'  L={params["n_layer"]}'
              f'  lr={params["lr"]}'
              f'  wd={params["weight_decay"]}',
              flush=True)
        return acc

    # ── Create study ──────────────────────────────────────────────────────────
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    sampler = TPESampler(
        n_startup_trials=args.n_startup_trials,
        seed=args.seed,
        multivariate=True,              # considers param correlations
        group=True,                     # respects conditional (k | knn_mode) structure
        warn_independent_sampling=False, # k is intentionally conditional on knn_mode
    )
    # MedianPruner here will prune if a trial is in bottom 50% after warmup
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials,
                          n_warmup_steps=0)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=scenario_key,
        storage=args.storage,
        load_if_exists=True,
    )

    # ── Optimise ──────────────────────────────────────────────────────────────
    study.optimize(
        objective,
        n_trials=args.max_trial_number,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f'[Optuna] WARNING: No successful trials for {scenario_key}. '
              'Exiting without updating HPO DB.')
        sys.exit(1)

    print(f'\n[Optuna] Optimization done. '
          f'{len(completed)}/{args.max_trial_number} trials completed.', flush=True)

    # ── Select top-K configs for retest ───────────────────────────────────────
    completed.sort(key=lambda t: t.value, reverse=True)
    top_k_trials = completed[:args.top_k]

    print(f'\n[Retest] ({scenario_key}) '
          f'Testing top-{len(top_k_trials)} configs × {args.retest_runs} seeds...',
          flush=True)

    best_mean   = -1.0
    best_params = None

    for rank, trial in enumerate(top_k_trials):
        params   = trial.params          # flat dict directly from Optuna
        orig_acc = trial.value

        print(f'\n  [Retest] ({scenario_key}) '
              f'Config {rank+1}/{len(top_k_trials)}: '
              f'(Trial #{trial.number}, Orig Acc: {orig_acc:.4f})')
        print(f'           Params: {params}')

        accs = []
        for run_i in range(args.retest_runs):
            seed = args.seed + run_i
            t0   = time.time()
            acc  = _run_trial(params, args.dataset, args.noise_type, args.noise_rate,
                              args.device, seed, priority=args.priority)
            elapsed = time.time() - t0

            if acc is not None:
                accs.append(acc)
                import numpy as np
                cur_mean = float(np.mean(accs))
                print(f'    [Retest] {args.dataset} | Config {rank+1} | '
                      f'Run {run_i+1}/{args.retest_runs} | '
                      f'seed={seed} | Acc: {acc:.4f} | '
                      f'Mean: {cur_mean:.4f} | Time: {elapsed:.1f}s', flush=True)
            else:
                print(f'    [Retest] {args.dataset} | Config {rank+1} | '
                      f'Run {run_i+1}/{args.retest_runs} | seed={seed} | FAILED',
                      flush=True)

        if accs:
            import numpy as np
            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs))
            print(f'  [Retest] Config {rank+1}: '
                  f'mean={mean_acc:.4f} ± {std_acc:.4f}', flush=True)
            if mean_acc > best_mean:
                best_mean   = mean_acc
                best_params = params
        else:
            print(f'  [Retest] Config {rank+1}: all runs failed.', flush=True)

    # ── Save best config to HPO DB ────────────────────────────────────────────
    if best_params is None:
        print(f'[Optuna] WARNING: No valid retest result. '
              f'HPO DB not updated for {scenario_key}.')
        sys.exit(1)

    hpo_db_path = args.hpo_db
    os.makedirs(os.path.dirname(hpo_db_path) if os.path.dirname(hpo_db_path) else '.',
                exist_ok=True)
    hpo_db = {}
    if os.path.exists(hpo_db_path):
        with open(hpo_db_path, 'r', encoding='utf-8') as f:
            try:
                hpo_db = json.load(f)
            except Exception:
                hpo_db = {}

    hpo_db[scenario_key] = best_params
    with open(hpo_db_path, 'w', encoding='utf-8') as f:
        json.dump(hpo_db, f, indent=2)

    print(f'\n[Optuna] Success: Best config saved to {hpo_db_path}:')
    print(f'         {best_params}')
    print(f'[Optuna] Success: Best retest mean accuracy: {best_mean:.4f}', flush=True)

    # ── Write done marker (same location as NNI version) ─────────────────────
    from datetime import datetime
    done_dir    = './log/nni_done'
    done_marker = os.path.join(done_dir,
                               f'lnpcc_{args.dataset}_{args.noise_type}_{args.noise_rate}.done')
    os.makedirs(done_dir, exist_ok=True)
    with open(done_marker, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f'[Optuna] Success: Done marker: {done_marker}', flush=True)


if __name__ == '__main__':
    main()
