import json
import numpy as np

with open('log/hpo_db_instance.json', 'r') as f:
    db = json.load(f)

dataset = 'amazon-ratings'
methods = {}

for key, entry in db.items():
    if dataset in key and 'retest_avg' in entry:
        parts = key.split('_')
        method = parts[0]
        if method not in methods:
            methods[method] = {'acc': [], 'wall': [], 'cpu': [], 'gpu': [], 'wait': []}
        
        methods[method]['acc'].append(entry['retest_avg'])
        finals = entry.get('all_final_results', [])
        if finals:
            methods[method]['wall'].append(np.mean([r.get('wall_s', 0) for r in finals]))
            methods[method]['cpu'].append(np.mean([r.get('cpu_s', 0) for r in finals]))
            methods[method]['gpu'].append(np.mean([r.get('gpu_s', 0) for r in finals]))
            methods[method]['wait'].append(np.mean([r.get('wait_s', 0) for r in finals]))

print(f"{'Method':<12} | {'Acc':<8} | {'Wall(s)':<8} | {'CPU(s)':<8} | {'GPU(s)':<8} | {'Wait(s)':<8}")
print("-" * 65)
for m in sorted(methods.keys()):
    d = methods[m]
    print(f"{m:<12} | {np.mean(d['acc']):<8.4f} | {np.mean(d['wall']):<8.1f} | {np.mean(d['cpu']):<8.1f} | {np.mean(d['gpu']):<8.1f} | {np.mean(d['wait']):<8.1f}")
