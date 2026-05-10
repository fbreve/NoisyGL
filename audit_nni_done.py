"""
audit_nni_done.py
Identifies and cleans up "ghost" HPO markers and sub-optimal Phase 2 partials.
A marker is considered "bad" if it exists in log/nni_done but has no optimized
entry in log/hpo_db.json.

Usage:
    python audit_nni_done.py           # Report missing entries
    python audit_nni_done.py --delete  # Clean up markers and partials
"""
import os
import json
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help='Delete bad markers and partials')
    args = parser.parse_args()

    hpo_db_path = 'log/hpo_db.json'
    nni_done_dir = 'log/nni_done'
    partial_dir = 'log/partial'

    # 1. Load HPO database
    hpo_db = {}
    if os.path.exists(hpo_db_path):
        with open(hpo_db_path, 'r', encoding='utf-8') as f:
            try:
                hpo_db = json.load(f)
            except Exception as e:
                print(f"Error loading HPO DB: {e}")
                return

    print(f"Loaded {len(hpo_db)} entries from {hpo_db_path}")

    # 2. Get all .done markers
    done_files = glob.glob(os.path.join(nni_done_dir, "*.done"))
    print(f"Found {len(done_files)} NNI markers in {nni_done_dir}")

    bad_scenarios = []
    
    # scenario keys in hpo_db: dataset_noiseType_noiseRate
    # filenames: lnpcc_dataset_noiseType_noiseRate.done
    
    for df in done_files:
        basename = os.path.basename(df)
        if not basename.startswith('lnpcc_') or not basename.endswith('.done'):
            continue
            
        # extract dataset_noiseType_noiseRate
        scenario_key = basename.replace('lnpcc_', '').replace('.done', '')
        
        if scenario_key not in hpo_db:
            bad_scenarios.append({
                'key': scenario_key,
                'marker': df,
                'partial': os.path.join(partial_dir, f"partial_lnpcc_{scenario_key}.csv")
            })

    if not bad_scenarios:
        print("\n[OK] No bad markers found. All 'done' scenarios have HPO data.")
        return

    print(f"\n[BAD] Found {len(bad_scenarios)} 'done' markers with MISSING HPO data:")
    for item in bad_scenarios:
        print(f"  - {item['key']}")
        if os.path.exists(item['partial']):
            print(f"    (Found Phase 2 partial: {os.path.basename(item['partial'])})")

    if args.delete:
        print(f"\n[CLEANUP] Deleting {len(bad_scenarios)} groups of files...")
        deleted_count = 0
        for item in bad_scenarios:
            # Delete marker
            if os.path.exists(item['marker']):
                os.remove(item['marker'])
                print(f"  Deleted marker:  {os.path.basename(item['marker'])}")
            
            # Delete partial if exists
            if os.path.exists(item['partial']):
                os.remove(item['partial'])
                print(f"  Deleted partial: {os.path.basename(item['partial'])}")
            
            deleted_count += 1
        
        print(f"\nCleanup finished. Deleted {deleted_count} scenarios.")
        print("\n>>> IMPORTANT: Run Phase 1 (NNI) again to complete the missing optimizations.")
        print(">>> After that, Phase 2 will automatically re-run those scenarios with optimized parameters.")
    else:
        print(f"\nRun with --delete to remove these {len(bad_scenarios)} bad markers and partials.")

if __name__ == "__main__":
    main()
