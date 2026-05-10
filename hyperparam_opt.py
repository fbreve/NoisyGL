import os
import sys
import time
import json
import argparse
import subprocess
import csv
from nni.experiment import Experiment
from utils.tools import load_conf, save_conf

os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# --- Helper functions ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho', 'dblp',
                                 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'])
    parser.add_argument('--method', type=str, default='lnpcc',
                        choices=['gcn', 'lnpcc', 'gin', 'smodel', 'jocor', 'coteaching', 'apl',
                                 'sce', 'forward', 'backward', 'lcat', 'tss', 'nrgnn', 'rtgnn',
                                 'cp', 'unionnet', 'cgnn', 'crgnn', 'clnode', 'rncgln', 'pignn',
                                 'dgnn', 'r2lp'])
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['clean', 'uniform', 'pair', 'random', 'instance'])
    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--max_trial_number', type=int, default=200)
    parser.add_argument('--trial_concurrency', type=int, default=1)
    parser.add_argument('--tuner', type=str, default='TPE')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--update_config', type=bool, default=True)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--retest_runs', type=int, default=10)

    args = parser.parse_args()

    scenario_key = f"{args.dataset}_{args.noise_type}_{args.noise_rate}"
    nni_timing_log = f"log/nni_timing_{args.dataset}_{args.noise_type}_{args.noise_rate}.log"

    experiment = Experiment('local')
    command = 'python single_exp.py'
    for k, v in sorted(vars(args).items()):
        if k in ['dataset', 'noise_type', 'noise_rate', 'device', 'seed']:
            command += ' --' + k + '=' + str(v)
    command += ' --method=' + args.method
    experiment.config.trial_command = command
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space_file = './config/_search_space/' + args.method + '.json'
    experiment.config.tuner.name = args.tuner
    experiment.config.assessor.name = 'Medianstop'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = args.max_trial_number
    experiment.config.trial_concurrency = args.trial_concurrency

    # ── Run with port retry ──────────────────────────────────────────────
    max_port_retries = 50
    current_port = args.port
    success = False
    print(f"\n[NNI] Scenario: {scenario_key} (Port: {current_port}, Max trials: {args.max_trial_number})")

    for attempt in range(max_port_retries):
        try:
            experiment.start(current_port, debug=False)
            success = True
            break
        except Exception as e:
            # Only print if we are not at the very last attempt or if it's a real error
            if attempt < max_port_retries - 1:
                print(f"  [NNI] Port {current_port} unavailable, trying {current_port + 1}...")
            current_port += 1

    if not success:
        print(f"  [NNI] ERROR: Could not start NNI experiment for {scenario_key}")
        sys.exit(1)

    # ── Monitor progress and tail the timing log ───────────────────────
    print(f"  [NNI] Optimization started. Watching for trial results...", flush=True)

    last_finished = 0
    timing_log_pos = 0  # byte offset for tail-like reading
    monitor_best_acc = 0.0

    while experiment.get_status() not in ['DONE', 'STOPPED', 'ERROR']:
        try:
            # 1. Tail-read the nni_timing log (most immediate source of truth)
            if os.path.exists(nni_timing_log):
                with open(nni_timing_log, 'r', encoding='utf-8') as f:
                    f.seek(timing_log_pos)
                    new_lines = f.read()
                    timing_log_pos = f.tell()
                if new_lines.strip():
                    for line in new_lines.strip().splitlines():
                        print(f"  [Trial] {line}", flush=True)
                        # Extract Acc from: Trial: ... | Acc: 0.1234 | ...
                        import re
                        m_acc = re.search(r"Acc:\s*([\d\.]+)", line)
                        if m_acc:
                            val = float(m_acc.group(1))
                            if val > monitor_best_acc:
                                monitor_best_acc = val

            # 2. Monitor NNI trial status
            trials = experiment.list_trial_jobs()
            finished = len([t for t in trials if t.status in
                            ['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED']])
            if finished > last_finished:
                best_str = f"{monitor_best_acc:.4f}" if monitor_best_acc > 0 else "N/A"
                print(f"  [NNI Progress] {scenario_key}: {finished}/{args.max_trial_number} trials. "
                      f"Best Acc so far: {best_str}", flush=True)
                last_finished = finished
        except Exception:
            pass
        time.sleep(10)

    print(f"  [NNI] Optimization finished for {scenario_key}.", flush=True)

    # ── Flush remaining timing log lines ─────────────────────────────────
    if os.path.exists(nni_timing_log):
        with open(nni_timing_log, 'r', encoding='utf-8') as f:
            f.seek(timing_log_pos)
            remaining = f.read()
        if remaining.strip():
            for line in remaining.strip().splitlines():
                print(f"  [Trial] {line}", flush=True)

    # ── Gather top-K trials for retest ────────────────────────────────────
    def _parse_log_for_top_k(log_path, k):
        """Parse the nni_timing log for the best k trials."""
        if not os.path.exists(log_path):
            return []
        
        trials = []
        import re, json
        # More flexible regex to capture ID, Acc and Params
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Better regex to catch ID, Acc and Params
                m_id = re.search(r"Trial:\s*(\w+)", line)
                m_acc = re.search(r"Acc:\s*([\d\.]+)", line)
                m_params = re.search(r"Params:\s*(\{.*\})", line)
                
                if m_id and m_acc and m_params:
                    try:
                        acc = float(m_acc.group(1))
                        params = json.loads(m_params.group(1))
                        trials.append({'id': m_id.group(1), 'acc': acc, 'params': params})
                    except:
                        continue
        
        # Sort by accuracy descending
        trials.sort(key=lambda x: x['acc'], reverse=True)
        return trials[:k]

    def _trial_best_metric(t):
        """Return best default metric across all finalMetricData entries."""
        vals = []
        for m in t.finalMetricData:
            try:
                # NNI v3+ sometimes returns a JSON string, others a dict.
                import json
                data = m.data
                if isinstance(data, str):
                    try: data = json.loads(data)
                    except: pass
                
                v = data.get('default', None) if isinstance(data, dict) else data
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
        return max(vals) if vals else 0.0

    # 1. Try strategy A: Read from our local log (most reliable)
    top_configs = []
    print(f"\n  [NNI] Selecting top-{args.top_k} from timing log: {nni_timing_log}")
    log_top = _parse_log_for_top_k(nni_timing_log, args.top_k)
    if log_top:
        for item in log_top:
            top_configs.append((item['acc'], item['params'], item['id']))
            print(f"    - Found Trial {item['id']} in log with Acc: {item['acc']:.4f}")
    
    # 2. Try strategy B: Fallback to NNI API if log is empty or we need more
    if len(top_configs) < args.top_k:
        print(f"  [NNI] Fallback: Querying NNI API for more trials...")
        try:
            all_trials = experiment.list_trial_jobs()
            succeeded = [t for t in all_trials if t.status == 'SUCCEEDED']
            # Avoid duplicates if we already got them from the log
            existing_ids = {c[2] for c in top_configs}
            succeeded = [t for t in succeeded if t.id not in existing_ids]
            succeeded.sort(key=_trial_best_metric, reverse=True)
            
            for t in succeeded[:args.top_k - len(top_configs)]:
                acc = _trial_best_metric(t)
                try:
                    params = t.hyperParameters[0].parameters if t.hyperParameters else {}
                except:
                    params = {}
                top_configs.append((acc, params, t.id))
                print(f"    - Added Trial {t.id} from NNI API with Acc: {acc:.4f}")
        except Exception as e:
            print(f"  [NNI] Error querying API: {e}")

    experiment.stop()
    print(f"  [NNI] Experiment stopped.", flush=True)

    if not top_configs:
        print(f"  [NNI] WARNING: No successful trials found for {scenario_key}. Skipping retest.")
        return

    # ── Retest top-K configs ──────────────────────────────────────────────
    print(f"\n  [Retest] ({scenario_key}) Testing top-{len(top_configs)} configs × {args.retest_runs} seeds...")
    
    # Sequential retest to avoid memory and thread conflicts on machines with many parallel experiments
    best_mean = -1.0
    best_params = None

    for rank, (orig_acc, params, tid) in enumerate(top_configs):
        print(f"\n  [Retest] ({scenario_key}) Config {rank+1}/{len(top_configs)}: (Trial {tid}, Orig Acc: {orig_acc:.4f})")
        print(f"           Params: {params}")
        accs = []
        params_json = json.dumps(params)

        for run_i in range(args.retest_runs):
            seed = args.seed + run_i
            cmd = [
                sys.executable, 'single_exp.py',
                '--dataset',    args.dataset,
                '--method',     args.method,
                '--noise_type', args.noise_type,
                '--noise_rate', str(args.noise_rate),
                '--device',     args.device,
                '--seed',       str(seed),
                '--params_json', params_json,
            ]
            
            import time as _time
            t_start = _time.time()
            try:
                # Sequential retest via subprocess.run
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                elapsed = _time.time() - t_start
                
                # Parse FINAL_RESULT from stdout
                import re, ast
                acc = None
                match = re.search(r"FINAL_RESULT:\s*(\{.*?\})", result.stdout)
                if match:
                    try:
                        res_dict = ast.literal_eval(match.group(1))
                        # Try to get 'test', if not present get value directly if dict or whatever comes
                        acc = float(res_dict.get('test', 0))
                    except:
                        pass
                
                if result.returncode == 0 and acc is not None:
                    accs.append(acc)
                    import numpy as np
                    cur_mean = float(np.mean(accs))
                    print(f"    [Retest] {args.dataset} | Config {rank+1}/3 | Trial {tid} | Run {run_i+1}/{args.retest_runs} | Seed {seed} | Acc: {acc:.4f} | Mean: {cur_mean:.4f} | Time: {elapsed:.1f}s", flush=True)
                else:
                    msg = "parse error" if result.returncode == 0 else f"CRASHED (code {result.returncode})"
                    print(f"    [Retest] {args.dataset} | Config {rank+1}/3 | Trial {tid} | Run {run_i+1}/{args.retest_runs} | Seed {seed} | ? {msg}", flush=True)
                    if result.returncode != 0:
                        if result.stdout:
                            print("      [STDOUT Last Bytes]:\n" + "\n".join(result.stdout.strip().splitlines()[-5:]), flush=True)
                        if result.stderr:
                            print("      [STDERR Last Bytes]:\n" + "\n".join(result.stderr.strip().splitlines()[-5:]), flush=True)
                    elif result.stderr:
                        # Show last line of error to help with diagnosis
                        last_err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Unknown error"
                        print(f"      [Last Error]: {last_err}", flush=True)
            except Exception as e:
                print(f"    [Retest] {args.dataset} | Config {rank+1}/3 | Trial {tid} | Run {run_i+1}/{args.retest_runs} | Seed {seed} | ? ERROR: {e}", flush=True)

        if accs:
            import numpy as np
            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs))
            print(f"  [Retest] Config {rank+1}: mean={mean_acc:.4f} ± {std_acc:.4f}", flush=True)
            if mean_acc > best_mean:
                best_mean = mean_acc
                best_params = params
        else:
            print(f"  [Retest] Config {rank+1}: all runs failed.", flush=True)

    # ── Save best config to centralized HPO DB ────────────────────────────
    if best_params is not None:
        hpo_db_path = 'log/hpo_db.json'
        os.makedirs('log', exist_ok=True)
        hpo_db = {}
        if os.path.exists(hpo_db_path):
            with open(hpo_db_path, 'r', encoding='utf-8') as f:
                try: hpo_db = json.load(f)
                except: hpo_db = {}

        hpo_db[scenario_key] = best_params
        with open(hpo_db_path, 'w', encoding='utf-8') as f:
            json.dump(hpo_db, f, indent=2)
        print(f"\n  [NNI] ✓ Best config saved to {hpo_db_path}: {best_params}", flush=True)
        print(f"  [NNI] ✓ Best retest mean accuracy: {best_mean:.4f}", flush=True)
    else:
        print(f"  [NNI] WARNING: No valid retest result. HPO DB not updated for {scenario_key}.")
        sys.exit(1)


if __name__ == '__main__':
    main()
