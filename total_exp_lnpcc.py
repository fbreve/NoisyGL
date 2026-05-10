"""
total_exp_lnpcc.py
Runs the NoisyGL benchmark for GCN and the unified LN-PCC model.

Usage examples:

  # Quick smoke test — Cora, uniform noise 30%, 2 runs
  python total_exp_lnpcc.py --datasets cora --noise_type uniform --noise_rate 0.3 --runs 2 --skip_hpo

  # Full benchmark — all datasets, all noise types (skip NNI, use pre-tuned YAMLs)
  python total_exp_lnpcc.py --all_datasets --all_noise --skip_hpo

  # Phase 1 only — HPO optimization per dataset (200 trials each)
  python total_exp_lnpcc.py --all_datasets --hpo_only --optimize_trials 200

  # Phase 2 — benchmark using already-tuned YAMLs
  python total_exp_lnpcc.py --all_datasets --all_noise --skip_hpo

Notes:
  - Results are saved to ./log/lnpcc_results_<timestamp>.csv
  - When called from run_lnpcc_parallel.py, each worker runs one (dataset × noise)
    combination and writes its own partial CSV.
"""

import argparse
import warnings
import os
import sys
import subprocess
import time
import csv
from datetime import datetime
import numpy as np

from utils.labelnoise import label_process
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.logger import MultiExpRecorder, ResultLogger

# ── Predictors ────────────────────────────────────────────────────────────────
from predictor.LNPCC_Predictor import lnpcc_Predictor
from predictor.GCN_Predictor import gcn_Predictor

# ── Variants ──────────────────────────────────────────────────────────────────
ALL_VARIANTS = ['lnpcc', 'gcn']
DEFAULT_VARIANTS = ['lnpcc']

ALL_DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'amazoncom', 'amazonpho', 'dblp', 'blogcatalog', 'flickr',
    'amazon-ratings', 'roman-empire',
]


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
        # Only print if not running as a worker (to avoid polluting output)
        if '--partial_csv' not in sys.argv:
            print(f'[priority] Process priority set to {level.upper()}', flush=True)
    except Exception as e:
        if '--partial_csv' not in sys.argv:
            print(f'[priority] Warning: could not set priority: {e}', flush=True)


# ── Single experiment ─────────────────────────────────────────────────────────

def run_single_exp(dataset, method_name, seed, noise_type, noise_rate, device, debug=False, hpo_db=None):
    setup_seed(seed)
    model_conf = load_conf(None, method_name, dataset.name)
    
    # Force uniformLabeled to False as requested
    if hasattr(model_conf, 'model'):
        model_conf.model['uniformLabeled'] = False

    # Apply HPO parameters if available for lnpcc
    if method_name == 'lnpcc' and hpo_db:
        scenario_key = f"{dataset.name}_{noise_type}_{noise_rate}"
        if scenario_key in hpo_db:
            params = hpo_db[scenario_key]
            print(f"  [HPO] Applying optimized params for {scenario_key}")
            
            def flatten_nni_params(params):
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
                
            flat_params = flatten_nni_params(params)
            for item, val in flat_params.items():
                if item in ['lr', 'weight_decay']:
                    model_conf.training[item] = val
                else:
                    model_conf.model[item] = val
        else:
            print(f"  [HPO] No entry for {scenario_key} in HPO DB, using YAML defaults.")

    dataset.noisy_label, modified_mask = label_process(
        labels=dataset.labels, features=dataset.feats,
        n_classes=dataset.n_classes,
        noise_type=noise_type, noise_rate=noise_rate,
        random_seed=seed, debug=debug,
    )

    incorrect_labeled_train_mask   = dataset.train_masks[np.isin(dataset.train_masks, modified_mask)]
    correct_labeled_train_mask     = dataset.train_masks[~np.isin(dataset.train_masks, modified_mask)]
    supervised_mask                = get_neighbors(dataset.adj, dataset.train_masks)
    incorrect_supervised_mask      = get_neighbors(dataset.adj, incorrect_labeled_train_mask)
    correct_supervised_mask        = get_neighbors(dataset.adj, correct_labeled_train_mask)
    unlabeled_incorrect_sup_mask   = dataset.test_masks[np.isin(dataset.test_masks, incorrect_supervised_mask)]
    unlabeled_correct_sup_mask     = dataset.test_masks[np.isin(dataset.test_masks, correct_supervised_mask)]
    unlabeled_unsupervised_mask    = dataset.test_masks[~np.isin(dataset.test_masks, supervised_mask)]

    model_conf.model['n_feat']    = dataset.dim_feats
    model_conf.model['n_classes'] = dataset.n_classes
    model_conf.training['debug']  = debug

    if method_name == 'gcn':
        predictor = gcn_Predictor(model_conf, dataset, device)
    else:
        predictor = lnpcc_Predictor(model_conf, dataset, device)

    original_result = predictor.train()

    _, correct_labeled_train_accuracy      = predictor.test(correct_labeled_train_mask)
    _, incorrect_labeled_train_accuracy    = predictor.test(incorrect_labeled_train_mask)
    _, incorrect_mislead_train_accuracy    = predictor.evaluate(
        predictor.noisy_label, incorrect_labeled_train_mask)
    _, unlabeled_unsupervised_accuracy     = predictor.test(unlabeled_unsupervised_mask)
    _, unlabeled_correct_sup_accuracy      = predictor.test(unlabeled_correct_sup_mask)
    _, unlabeled_incorrect_sup_accuracy    = predictor.test(unlabeled_incorrect_sup_mask)

    extended_result = dict(original_result)
    extended_result['correct_labeled_train_accuracy']           = correct_labeled_train_accuracy
    extended_result['incorrect_labeled_train_accuracy']         = incorrect_labeled_train_accuracy
    extended_result['incorrect_labeled_mislead_train_accuracy'] = incorrect_mislead_train_accuracy
    extended_result['unlabeled_correct_supervised_accuracy']    = unlabeled_correct_sup_accuracy
    extended_result['unlabeled_unsupervised_accuracy']          = unlabeled_unsupervised_accuracy
    extended_result['unlabeled_incorrect_supervised_accuracy']  = unlabeled_incorrect_sup_accuracy
    extended_result['total_time']                               = predictor.total_time

    return original_result, extended_result


# ── NNI optimization ──────────────────────────────────────────────────────────

# Dataset size heuristic: large datasets need smaller k for knn_mode='s'
_LARGE_DATASETS = {'amazon-ratings', 'roman-empire', 'flickr', 'blogcatalog'}

def _default_max_k_same(dataset_name):
    """Return a safe default max_k for knn_mode='s' based on dataset size.

    k=60 is empirically verified to complete without OOM/crash on all datasets
    including amazon-ratings. k=100 caused 0xC0000409 crashes on large graphs.
    """
    return 60  # k<=60 verified stable; k=100 caused STATUS_STACK_BUFFER_OVERRUN


def optimize_variant(variant, dataset_name, noise_type, noise_rate, device,
                     max_trials=200, top_k=3, retest_runs=10, hpo_db='log/hpo_db.json',
                     max_k_same=None, priority='normal'):

    """
    Runs Optuna hyperparameter optimization for the unified `lnpcc` model on one
    (dataset, noise_type, noise_rate) scenario, then saves the result to hpo_db.json.
    """
    import hyperparam_opt_optuna as hpo
    scenario_key = f"{dataset_name}_{noise_type}_{noise_rate}"
    if max_k_same is None:
        max_k_same = _default_max_k_same(dataset_name)
    print(f"\n[HPO] Optimizing lnpcc on {scenario_key} "
          f"(max_trials={max_trials}, max_k_same={max_k_same})")

    # Mock sys.argv so hyperparam_opt_optuna.main() can parse args
    old_argv = sys.argv
    sys.argv = [
        'hyperparam_opt_optuna.py',
        '--method',            'lnpcc',
        '--dataset',           dataset_name,
        '--noise_type',        noise_type,
        '--noise_rate',        str(noise_rate),
        '--device',            device,
        '--max_trial_number',  str(max_trials),
        '--top_k',             str(top_k),
        '--retest_runs',       str(retest_runs),
        '--hpo_db',            hpo_db,
        '--max_k_same',        str(max_k_same),
        '--priority',          priority,
    ]

    done_marker = os.path.join('./log/nni_done',
                               f'{variant}_{dataset_name}_{noise_type}_{noise_rate}.done')
    try:
        hpo.main()
        # Done marker is written by hyperparam_opt_optuna.main() itself,
        # but write it here as well in case of any edge-case exit path.
        os.makedirs('./log/nni_done', exist_ok=True)
        if not os.path.exists(done_marker):
            with open(done_marker, 'w') as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"  [HPO] Done marker saved: {done_marker}")
    except BaseException as e:
        print(f"  [HPO] FAILED for {scenario_key}: {e}")
        sys.exit(1)
    finally:
        sys.argv = old_argv


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='NoisyGL benchmark runner for GCN + unified LN-PCC'
    )
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--variants', type=str, nargs='+', default=DEFAULT_VARIANTS,
                        choices=ALL_VARIANTS,
                        help='Variants to run (default: lnpcc only).')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cora', 'citeseer', 'pubmed'],
                        choices=ALL_DATASETS)
    parser.add_argument('--all_datasets', action='store_true')
    parser.add_argument('--noise_type', type=str, nargs='+',
                        default=['clean', 'uniform', 'pair', 'random'],
                        choices=['clean', 'uniform', 'pair', 'random', 'instance'])
    parser.add_argument('--all_noise', action='store_true')
    parser.add_argument('--noise_rate', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--optimize', action='store_true',
                        help='Run Optuna HPO for lnpcc before benchmark runs')
    parser.add_argument('--hpo_only', action='store_true',
                        help='PHASE 1: Only run Optuna HPO, skip benchmark runs')
    parser.add_argument('--skip_hpo', action='store_true',
                        help='PHASE 2: Skip HPO, only run benchmark')
    parser.add_argument('--optimize_trials', type=int, default=200,
                        help='Max Optuna trials per HPO run (default 200)')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--retest_runs', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--partial_csv', type=str, default=None,
                        help='Write results only to this path (used by run_lnpcc_parallel.py)')
    parser.add_argument('--hpo_db', type=str, default='log/hpo_db.json',
                        help='Path to HPO results JSON database')
    parser.add_argument('--max_k_same', type=int, default=None,
                        help='Maximum k for knn_mode="s" (None=auto-detect by dataset). '
                             'Set lower (e.g. 15) for large datasets to prevent timeouts.')
    parser.add_argument('--priority', type=str, default='normal',
                        choices=['normal', 'below_normal', 'idle'],
                        help='Process priority (default: normal). '
                             'When run via run_lnpcc_parallel, priority is managed by the launcher.')
    return parser.parse_args()


def _write_partial_csv(our_results, method_names, path):
    """Rewrite the partial CSV with all results collected so far."""
    fieldnames = ['dataset', 'noise_rate'] + \
                 [f'{m}_{s}' for m in method_names for s in ('mean', 'std')]
    rows = {}
    for (method_name, data_name, noise_type, noise_rate), stats in our_results.items():
        noise_label = f'{noise_type}_{noise_rate}' if noise_type != 'clean' else 'clean_0.0'
        key = (data_name, noise_label)
        if key not in rows:
            rows[key] = {'dataset': data_name, 'noise_rate': noise_label}
        rows[key][f'{method_name}_mean'] = round(stats['mean'], 6) if not np.isnan(stats['mean']) else ''
        rows[key][f'{method_name}_std']  = round(stats['std'],  6) if not np.isnan(stats['std'])  else ''
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: (r['dataset'], r['noise_rate'])):
            writer.writerow(row)


def main():
    args = parse_args()
    set_priority(args.priority)
    warnings.filterwarnings('ignore')

    variants    = args.variants or DEFAULT_VARIANTS
    datasets    = ALL_DATASETS if args.all_datasets else args.datasets
    noise_types = ['clean', 'uniform', 'pair', 'random'] if args.all_noise \
                  else args.noise_type

    method_names = variants  # gcn stays 'gcn', lnpcc stays 'lnpcc'

    noise_list = []
    for nt in noise_types:
        if nt == 'clean':
            noise_list.append([0.0, 'clean'])
        else:
            for nr in args.noise_rate:
                noise_list.append([nr, nt])

    print('=' * 70)
    print('GCN + LN-PCC -- NoisyGL Benchmark (Unified)')
    print(f'Start          : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Variants       : {variants}')
    print(f'Datasets       : {datasets}')
    print(f'Noise          : {noise_list}')
    print(f'Runs           : {args.runs}  (seeds {args.seed}..{args.seed+args.runs-1})')
    print(f'Device         : {args.device}')
    print('=' * 70)

    # ── Optional HPO optimization ──────────────────────────────────────
    if args.optimize and not args.skip_hpo:
        if 'lnpcc' in variants:
            for ds in datasets:
                for nr, nt in noise_list:
                    job_key = f'lnpcc_{ds}_{nt}_{nr}'
                    done_marker = os.path.join('./log/nni_done', f'{job_key}.done')
                    if os.path.exists(done_marker):
                        print(f'  [HPO SKIP] {job_key} -- already optimized')
                        continue

                    # instance noise is not optimized yet, skip it for now
                    if nt == 'instance':
                        print(f'  [HPO SKIP] {job_key} -- instance noise skipped (no HPO)')
                        continue

                    optimize_variant('lnpcc', ds, nt, nr, args.device,
                                     max_trials=args.optimize_trials,
                                     top_k=args.top_k,
                                     retest_runs=args.retest_runs,
                                     hpo_db=args.hpo_db,
                                     max_k_same=args.max_k_same,
                                     priority=args.priority)

    if args.hpo_only:
        print('[total_exp] --hpo_only: skipping benchmark runs.')
        return

    # ── Setup result storage ───────────────────────────────────────────
    os.makedirs('./log', exist_ok=True)
    use_noisygl_logger = args.partial_csv is None
    result_recorder = ResultLogger(method_names, datasets, noise_list, args.runs) \
                      if use_noisygl_logger else None

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    our_csv_path = args.partial_csv if args.partial_csv else f'./log/lnpcc_results_{ts}.csv'
    our_results = {}

    # ── Load checkpoint ────────────────────────────────────────────────
    done_keys = set()
    if os.path.exists(our_csv_path):
        try:
            with open(our_csv_path, 'r', encoding='utf-8') as _f:
                for row in csv.DictReader(_f):
                    ds = row['dataset']
                    nl = row['noise_rate']
                    if nl == 'clean_0.0':
                        nt, nr = 'clean', 0.0
                    else:
                        parts = nl.rsplit('_', 1)
                        nt, nr = parts[0], float(parts[1])
                    for mn in method_names:
                        mean_key = f'{mn}_mean'
                        if mean_key in row and row[mean_key] != '':
                            done_keys.add((mn, ds, nt, nr))
                            our_results[(mn, ds, nt, nr)] = {
                                'mean': float(row[mean_key]),
                                'std':  float(row.get(f'{mn}_std', 0) or 0),
                            }
            if done_keys:
                print(f'[checkpoint] Loaded {len(done_keys)} completed entries from {our_csv_path}')
        except Exception as e:
            print(f'[checkpoint] Warning: could not read checkpoint: {e}')

    data_path = './data/'
    
    # ── Load HPO Database ──────────────────────────────────────────────
    hpo_db = {}
    if os.path.exists(args.hpo_db):
        import json
        try:
            with open(args.hpo_db, 'r', encoding='utf-8') as f:
                hpo_db = json.load(f)
            print(f'[HPO] Loaded {len(hpo_db)} optimized scenarios from {args.hpo_db}')
        except Exception as e:
            print(f'[HPO] Warning: could not load HPO DB: {e}')

    # ── Main experiment loop ───────────────────────────────────────────
    for data_name in datasets:
        all_done = all(
            (mn, data_name, nt, nr) in done_keys
            for mn in method_names
            for nr, nt in noise_list
        )
        if all_done:
            for nr, nt in noise_list:
                for method_name in method_names:
                    print(f'  [DONE] {method_name} | {data_name} | {nt} {nr} (checkpoint)')
            continue

        setup_seed(args.seed)
        data_conf = load_conf('./config/_dataset/' + data_name + '.yaml')
        data = Dataset(
            data_name, path=data_path,
            feat_norm   = data_conf.norm['feat_norm'],
            adj_norm    = data_conf.norm['adj_norm'],
            train_size  = data_conf.split['train_size'],
            val_size    = data_conf.split['val_size'],
            test_size   = data_conf.split['test_size'],
            train_percent              = data_conf.split['train_percent'],
            val_percent                = data_conf.split['val_percent'],
            test_percent               = data_conf.split['test_percent'],
            train_examples_per_class   = data_conf.split['train_examples_per_class'],
            val_examples_per_class     = data_conf.split['val_examples_per_class'],
            test_examples_per_class    = data_conf.split['test_examples_per_class'],
            add_self_loop              = data_conf.modify['add_self_loop'],
            from_npz                   = data_conf.modify['from_npz_largest_component'],
            device      = args.device,
            split_type  = data_conf.split['split_type'],
        )

        for noise_rate, noise_type in noise_list:
            # --- HARD FAILSAFE skips ---
            # Skip instance noise ONLY if it's coming from a bulk run (--all_noise)
            # This allows running it explicitly with --noise_type instance
            if noise_type == 'instance' and args.all_noise:
                print(f'  [HARD SKIP] {data_name} | instance noise skipped.')
                continue

            for method_name in method_names:
                # Same for baseline GCN: skip bulk runs, allow explicit
                # Skip GCN if not explicitly requested via --variants gcn
                if method_name == 'gcn' and 'gcn' not in args.variants:
                    print(f'  [HARD SKIP] {method_name} | {data_name} baseline skipped.')
                    continue

                if (method_name, data_name, noise_type, noise_rate) in done_keys:
                    print(f'  [DONE] {method_name} | {data_name} | {noise_type} {noise_rate} (checkpoint)')
                    continue

                conf_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
                conf_path = os.path.join(conf_dir, method_name, f'{method_name}_{data_name}.yaml')
                if not os.path.exists(conf_path):
                    print(f'\n  [ERROR] {method_name} | {data_name} -- config not found: {conf_path}')
                    continue

                print(f'\n{"="*60}')
                print(f'  {method_name} | {data_name} | {noise_type} {noise_rate}')
                print(f'{"="*60}')
                logger = MultiExpRecorder(runs=args.runs)
                acc_list = []

                successful   = 0
                attempt      = 0
                max_attempts = args.runs * 3
                while successful < args.runs and attempt < max_attempts:
                    seed = args.seed + attempt
                    try:
                        simple_result, total_result = run_single_exp(
                            data, method_name, seed=seed,
                            noise_type=noise_type, noise_rate=noise_rate,
                            device=args.device, debug=args.debug,
                            hpo_db=hpo_db
                        )
                        logger.results[successful] = []
                        logger.add_result(successful, total_result)
                        acc_list.append(simple_result['test'])
                        
                        import numpy as np
                        cur_mean = float(np.mean(acc_list))
                        elapsed = total_result.get('total_time', 0.0)
                        
                        print(f'    [{datetime.now().strftime("%H:%M:%S")}] {data_name} | {noise_type} {noise_rate} | '
                              f'{method_name} | run {successful+1:2d}/{args.runs} | seed={seed} | '
                              f'acc={simple_result["test"]:.4f} | mean={cur_mean:.4f} | time={elapsed:.2f}s', flush=True)
                        successful += 1
                    except Exception as e:
                        print(f'  run {successful+1:2d}/{args.runs} attempt {attempt+1} | '
                              f'seed={seed} | ERROR: {e} — retrying...')
                    attempt += 1

                if successful == 0:
                    print(f'  -> all attempts failed ({max_attempts} tries), skipping')
                    continue
                if successful < args.runs:
                    print(f'  -> WARNING: only {successful}/{args.runs} runs succeeded '
                          f'after {attempt} attempts; using partial results')
                    logger.results = [r for r in logger.results if len(r) > 0]

                valid_acc = [a for a in acc_list if not np.isnan(a)]
                total_results = logger.get_statistics()
                if result_recorder is not None:
                    result_recorder.dump_record(method_name, data_name, noise_type, noise_rate, total_results)

                mean_acc = float(np.mean(valid_acc))
                std_acc  = float(np.std(valid_acc))
                our_results[(method_name, data_name, noise_type, noise_rate)] = {
                    'mean': mean_acc, 'std': std_acc,
                }
                done_keys.add((method_name, data_name, noise_type, noise_rate))
                print(f'  -> mean={mean_acc:.4f} +/- {std_acc:.4f}')

                _write_partial_csv(our_results, method_names, our_csv_path)

    _write_partial_csv(our_results, method_names, our_csv_path)

    if use_noisygl_logger:
        print(f'\n{"="*70}')
        print('[OK] Done. Results saved to:')
        print(f'   Our CSV: {our_csv_path}')
    else:
        print(f'\n[OK] Partial CSV: {our_csv_path}')
    print(f'End: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
