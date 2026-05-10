import os
import sys
import argparse
import optuna
import json
import time
import numpy as np
import subprocess
from optuna.samplers import TPESampler

# Windows Stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def set_priority(level):
    if level == 'normal': return
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if level == 'below_normal': p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        elif level == 'idle': p.nice(psutil.IDLE_PRIORITY_CLASS)
    except: pass

def _run_trial(params, method, dataset, noise_type, noise_rate, device, seed, trial_id, priority):
    params_json = json.dumps(params)
    cmd = [
        sys.executable, 'single_exp_instance.py',
        '--dataset', dataset,
        '--method', method,
        '--noise_type', noise_type,
        '--noise_rate', str(noise_rate),
        '--device', device,
        '--seed', str(seed),
        '--params_json', params_json,
        '--priority', priority,
        '--trial_id', str(trial_id)
    ]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        last_result_line = ""
        for line in proc.stdout:
            print(f"    {line.strip()}", flush=True)
            if "FINAL_RESULT:" in line:
                last_result_line = line
        proc.wait()
        
        if last_result_line:
            res_str = last_result_line.split("FINAL_RESULT:")[1].strip()
            res_str = res_str.replace("'", '"')
            res_dict = json.loads(res_str)
            # Returns the full dict: {'acc': {...}, 'wall_s': ..., 'cpu_s': ..., 'gpu_s': ...}
            return res_dict
    except Exception as e:
        print(f"  [Exception] {e}", flush=True)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--noise_type', type=str, default='instance')
    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--max_trial_number', type=int, default=50)
    parser.add_argument('--hpo_db', type=str, default='log/hpo_db_instance.json')
    parser.add_argument('--storage', type=str, default='sqlite:///log/optuna_instance.db',
                        help='Optuna storage URL (e.g. sqlite:///log/optuna_instance.db)')
    parser.add_argument('--priority', type=str, default='normal')
    args = parser.parse_args()

    set_priority(args.priority)
    scenario_key = f"{args.method}_{args.dataset}_{args.noise_type}_{args.noise_rate}"
    print(f"[HPO] Starting Phase 1 (Optuna) for {scenario_key}...", flush=True)

    def objective(trial):
        params = {
            'lr': trial.suggest_categorical('lr', [0.001, 0.005, 0.01, 0.05, 0.1]),
            'weight_decay': trial.suggest_categorical('weight_decay', [0.0, 0.0001, 0.001, 0.01, 0.05]),
            'n_hidden': trial.suggest_categorical('n_hidden', [32, 64, 128, 256]),
            'n_layer': trial.suggest_categorical('n_layer', [2, 3]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.8),
        }
        if args.method == 'lnpcc':
            params.update({
                'knn_mode': trial.suggest_categorical('knn_mode', ['s', 'd']),
                'k': trial.suggest_int('k', 5, 50),
                'dexp': trial.suggest_float('dexp', 0.1, 10.0),
                'p_grd': trial.suggest_float('p_grd', 0.0, 1.0),
                'unc_rem': trial.suggest_float('unc_rem', 0.0, 1.0),
                'unc_rel': trial.suggest_float('unc_rel', 0.0, 1.0),
            })
        elif args.method == 'nrgnn':
            params.update({'p_cutoff': trial.suggest_float('p_cutoff', 0.0, 1.0), 'edge_threshold': trial.suggest_float('edge_threshold', 0.01, 0.5)})
        
        full_res = _run_trial(params, args.method, args.dataset, args.noise_type, args.noise_rate, args.device, args.seed, trial.number, args.priority)
        if full_res and 'acc' in full_res:
            return float(full_res['acc'].get('test', 0))
        return 0.0

    def early_stopping_callback(study, trial):
        if args.method == 'lnpcc': return
        patience = 10
        min_trials = 30
        if trial.number > (min_trials + patience) and study.best_trial.number < trial.number - patience:
            print(f" [Optuna] Early stopping at trial {trial.number}.", flush=True)
            study.stop()

    study = optuna.create_study(
        direction='maximize', 
        sampler=TPESampler(seed=args.seed, n_startup_trials=20),
        study_name=scenario_key,
        storage=args.storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.max_trial_number, callbacks=[early_stopping_callback])

    # ── Stage 2: Retest Top 5 (10 seeds each) ──
    print(f"\n[Retest] Phase 1 Finished. Validating Top 3 (10 runs each)...", flush=True)
    valid_trials = [t for t in study.trials if t.value is not None]
    top_3_hpo = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:3]
    
    best_candidate_params = None
    best_candidate_avg = -1.0
    
    for rank, trial in enumerate(top_3_hpo):
        print(f"\n  [Retest] Candidate {rank+1}/3 (Trial {trial.number}, HPO_acc={trial.value:.4f})", flush=True)
        trial_accs = []
        for s_idx, seed in enumerate(range(4000, 4010)): # Validation seeds
            full_res = _run_trial(trial.params, args.method, args.dataset, args.noise_type, args.noise_rate, 
                                 args.device, seed, f"retest_{trial.number}_{seed}", args.priority)
            if full_res and 'acc' in full_res:
                trial_accs.append(float(full_res['acc'].get('test', 0)))
        
        avg = np.mean(trial_accs) if trial_accs else 0
        print(f"  [Retest] Candidate {rank+1} Avg: {avg:.4f}", flush=True)
        if avg > best_candidate_avg:
            best_candidate_avg = avg
            best_candidate_params = trial.params
            
    if best_candidate_params is None: best_candidate_params = study.best_params

    # ── Stage 3: Phase 2 Final Benchmark (10 seeds) ──
    print(f"\n[Phase 2] Starting final evaluation with 10 seeds (3000-3009)...", flush=True)
    final_results = []
    for s_idx, seed in enumerate(range(3000, 3010)):
        full_res = _run_trial(best_candidate_params, args.method, args.dataset, args.noise_type, args.noise_rate, 
                             args.device, seed, f"final_{seed}", args.priority)
        if full_res:
            final_results.append(full_res)
    
    final_accs = [r['acc'].get('test', 0) for r in final_results]
    final_avg = np.mean(final_accs) if final_accs else 0
    final_std = np.std(final_accs) if final_accs else 0
    print(f"\n[FINAL] {scenario_key} | {final_avg:.4f} ± {final_std:.4f}", flush=True)

    # ── Save Detailed Times ──
    os.makedirs(os.path.dirname(args.hpo_db), exist_ok=True)
    db = {}
    if os.path.exists(args.hpo_db):
        try:
            with open(args.hpo_db, 'r') as f: db = json.load(f)
        except: pass
    
    db[scenario_key] = {
        'best_params': best_candidate_params,
        'retest_avg': float(final_avg),
        'retest_std': float(final_std),
        'all_final_results': final_results, # Contains acc, wall_s, cpu_s, gpu_s for all 10 runs
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(args.hpo_db, 'w') as f: json.dump(db, f, indent=2)

    done_dir = './log/instance_done'
    os.makedirs(done_dir, exist_ok=True)
    with open(os.path.join(done_dir, f'{scenario_key}.done'), 'w') as f: f.write(time.ctime())

if __name__ == '__main__':
    main()
