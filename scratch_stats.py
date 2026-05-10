import csv
from collections import defaultdict
import os

def calculate_stats():
    csv_path = 'log/instance_trials_all.csv'
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    data = defaultdict(lambda: {'wall': [], 'cpu': [], 'gpu': [], 'count': 0})
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = row['method']
                try:
                    # Tenta ler wall_s
                    data[m]['wall'].append(float(row.get('wall_s', 0)))
                    
                    # Tenta ler CPU (pode estar como cpu_s ou pcc_cpu_s)
                    cpu_val = row.get('cpu_s') or row.get('pcc_cpu_s') or 0
                    data[m]['cpu'].append(float(cpu_val))
                    
                    # Tenta ler GPU (pode estar como gpu_s ou gcn_gpu_s)
                    gpu_val = row.get('gpu_s') or row.get('gcn_gpu_s') or 0
                    data[m]['gpu'].append(float(gpu_val))
                    
                    data[m]['count'] += 1
                except (ValueError, TypeError):
                    continue

        if not data:
            print("Nenhum dado válido encontrado no CSV.")
            return

        print(f"{'Método':<10} | {'Trials':<8} | {'Avg Wall (s)':<12} | {'Avg CPU (s)':<12} | {'Avg GPU (s)':<12}")
        print("-" * 65)
        for m, v in sorted(data.items()):
            n = len(v['wall'])
            if n == 0: continue
            avg_w = sum(v['wall']) / n
            avg_c = sum(v['cpu']) / n
            avg_g = sum(v['gpu']) / n
            print(f"{m:<10} | {v['count']:8d} | {avg_w:12.2f} | {avg_c:12.2f} | {avg_g:12.2f}")
            
    except Exception as e:
        print(f"Erro ao processar CSV: {e}")

if __name__ == "__main__":
    calculate_stats()
