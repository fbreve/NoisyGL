import json
from collections import defaultdict

file_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\hpo_db_instance.json'

with open(file_path, 'r') as f:
    data = json.load(f)

stats = defaultdict(lambda: {'cpu': [], 'gpu': []})

for key, val in data.items():
    # Key example: "gcn_amazon-ratings_instance_0.2"
    parts = key.split('_')
    if len(parts) < 2:
        continue
    
    method = parts[0]
    dataset = parts[1]
    
    results = val.get('all_final_results', [])
    for res in results:
        cpu = res.get('cpu_s')
        gpu = res.get('gpu_s')
        if cpu is not None and gpu is not None:
            stats[(dataset, method)]['cpu'].append(cpu)
            stats[(dataset, method)]['gpu'].append(gpu)

print("| Dataset | Algoritmo | CPU Time (avg) | GPU Time (avg) | Total Ativo |")
print("| :--- | :--- | :---: | :---: | :---: |")

# Sort datasets then methods
sorted_keys = sorted(stats.keys())

for ds, m in sorted_keys:
    cpu_list = stats[(ds, m)]['cpu']
    gpu_list = stats[(ds, m)]['gpu']
    
    if not cpu_list:
        continue
        
    avg_cpu = sum(cpu_list) / len(cpu_list)
    avg_gpu = sum(gpu_list) / len(gpu_list)
    total = avg_cpu + avg_gpu
    
    print(f"| {ds} | {m.upper()} | {avg_cpu:7.2f}s | {avg_gpu:7.2f}s | {total:7.2f}s |")
