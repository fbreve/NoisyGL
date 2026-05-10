"""
fix_partials_nni.py
Remove entries for datasets that had failed NNI optimization (acc=0)
from all partial_*_nni.csv files, so they get re-run with correct params.

Usage: python fix_partials_nni.py [--partial_dir ./log/partial] [--dry_run]
"""
import csv
import os
import sys
import argparse
import shutil
from datetime import datetime

FAILED_DATASETS = ['amazoncom', 'amazonpho', 'dblp']

parser = argparse.ArgumentParser()
parser.add_argument('--partial_dir', default='./log/partial')
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

partial_dir = args.partial_dir
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = os.path.join(partial_dir, f'backup_{ts}')

nni_files = [f for f in os.listdir(partial_dir)
             if f.startswith('partial_') and f.endswith('_nni.csv')]

if not nni_files:
    print(f"No *_nni.csv files found in {partial_dir}")
    sys.exit(0)

print(f"Found {len(nni_files)} _nni partial files")
print(f"Will remove rows for datasets: {FAILED_DATASETS}")
if args.dry_run:
    print("[DRY RUN] No files will be modified\n")

for fname in sorted(nni_files):
    fpath = os.path.join(partial_dir, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = csv.DictReader(open(fpath, encoding='utf-8')).fieldnames

    kept   = [r for r in rows if r['dataset'] not in FAILED_DATASETS]
    removed = [r for r in rows if r['dataset'] in FAILED_DATASETS]

    if not removed:
        print(f"  {fname}: nothing to remove")
        continue

    removed_ds = list({r['dataset'] for r in removed})
    print(f"  {fname}: removing {len(removed)} rows for {removed_ds}")

    if not args.dry_run:
        # Backup first
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(fpath, os.path.join(backup_dir, fname))
        # Rewrite
        with open(fpath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

if not args.dry_run:
    print(f"\nBackups saved to: {backup_dir}")
    print("Done. Now delete the .done markers and restore default YAMLs, then re-run with --resume.")
