"""
total_exp_instance.py
=====================
Final benchmark runner for the Instance Noise Experiment.
Evaluates all 5 methods using the best params found in hpo_db_instance.json.

Usage:
  python total_exp_instance.py --runs 10
"""
import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime
from hyperparam_opt_instance import _run_trial

DATASETS = [
    'cora', 'citeseer', 'pubmed',
    'amazoncom', 'amazonpho', 'dblp',
    'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'
]
METHODS = ['lnpcc', 'gcn', 'nrgnn', 'pignn', 'cp']
NOISE_TYPE = 'instance'
NOISE_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--hpo_db', type=str, default='log/hpo_db_instance.json')
    parser.add_argument('--out_csv', type=str, default=None)
    parser.add_argument('--priority', type=str, default='normal', choices=['normal', 'below_normal', 'idle'])
    parser.add_argument('--resume', action='store_true', help='Skip already evaluated methods')
    args = parser.parse_args()

    if not args.out_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_csv = f'log/instance_results_{timestamp}.csv'
    
    os.makedirs('log', exist_ok=True)
    
    # Load HPO database
    hpo_db = {}
    if os.path.exists(args.hpo_db):
        with open(args.hpo_db, 'r', encoding='utf-8') as f:
            hpo_db = json.load(f)
    else:
        print(f"WARNING: HPO DB {args.hpo_db} not found. Running with defaults.")

    # Load existing results for resume
    done_set = set()
    if args.resume and os.path.exists(args.out_csv):
        import csv
        with open(args.out_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 3:
                    # method, ds, rate
                    done_set.add(f"{row[0]}_{row[1]}_{NOISE_TYPE}_{row[2]}")

    import csv
    mode = 'a' if (args.resume and os.path.exists(args.out_csv)) else 'w'
    with open(args.out_csv, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(['Method', 'Dataset', 'Noise_Rate', 'Mean_Acc', 'Std_Acc'])

        for ds in DATASETS:
            for rate in NOISE_RATES:
                for method in METHODS:
                    scenario_key = f"{method}_{ds}_{NOISE_TYPE}_{rate}"
                    if scenario_key in done_set:
                        print(f"\nSkipping {scenario_key} (already evaluated).")
                        continue

                    params = hpo_db.get(scenario_key, {})
                    
                    print(f"\nEvaluating {scenario_key} | {args.runs} runs...")
                    accs = []
                    for r in range(args.runs):
                        seed = args.seed + r
                        acc = _run_trial(params, method, ds, NOISE_TYPE, rate, 
                                         args.device, seed, priority=args.priority)
                        if acc is not None:
                            accs.append(acc)
                            print(f"  Run {r+1}/{args.runs}: {acc:.4f}")
                        else:
                            print(f"  Run {r+1}/{args.runs}: FAILED")
                    
                    if accs:
                        mean_acc = np.mean(accs)
                        std_acc = np.std(accs)
                        print(f"  RESULT: {mean_acc:.4f} +/- {std_acc:.4f}")
                        writer.writerow([method, ds, rate, f"{mean_acc:.4f}", f"{std_acc:.4f}"])
                        f.flush()

    print(f"\n[OK] Benchmarking complete. Results saved to {args.out_csv}")

if __name__ == '__main__':
    main()
