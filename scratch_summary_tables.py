import openpyxl, csv, re
import numpy as np

XLSX_PATH = r'results/NoisyGL.xlsx'
CSV_PATH = r'log/lnpcc_results_20260421_172348.csv'
INSTANCE_CSV_PATH = r'log/instance_final_results.csv'

all_results = {}
def add_result(method, dataset, ntype, rate, data):
    if data is None: return
    if method not in all_results: all_results[method] = {}
    if dataset not in all_results[method]: all_results[method][dataset] = {}
    if ntype not in all_results[method][dataset]: all_results[method][dataset][ntype] = {}
    all_results[method][dataset][ntype][rate] = data

def parse_noise_label_excel(noise_raw):
    noise_raw = str(noise_raw).strip().lower()
    if 'clean' in noise_raw: return 'clean', 0.0
    m = re.search(r'(\d+)\s*%?\s+([a-z-]+)', noise_raw)
    if m:
        rate = float(m.group(1)) / 100.0
        ntype = m.group(2)
        if 'asym' in ntype: ntype = 'random'
        return ntype, rate
    return None, None

def parse_val_with_std(val_str):
    if not val_str or str(val_str).lower() == 'nan': return None
    nums = re.findall(r'[\d.]+', str(val_str))
    if len(nums) >= 2: return {'mean': float(nums[0]), 'std': float(nums[1])}
    elif len(nums) == 1: return {'mean': float(nums[0]), 'std': 0.0}
    return None

EXCEL_DS_MAP = {
    'cora': 'cora', 'citeseer': 'citeseer', 'pubmed': 'pubmed',
    'amazon-c': 'amazoncom', 'amazon-p': 'amazonpho', 'dblp': 'dblp',
    'blogcatalog': 'blogcatalog', 'flickr': 'flickr', 'amz-rat.': 'amazon-ratings',
    'roman-emp.': 'roman-empire', 'a-computers': 'amazoncom', 'a-photo': 'amazonpho',
    'a-photos': 'amazonpho', 'a-ratings': 'amazon-ratings', 'empire': 'roman-empire',
    'amazonpho': 'amazonpho', 'amazoncom': 'amazoncom',
}

wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
sheet = wb.active
rows = list(sheet.iter_rows(values_only=True))
header_row_idx = -1
for i in range(20):
    if len(rows[i]) > 2 and str(rows[i][2]).strip() == 'Dataset':
        header_row_idx = i
        break

methods_found = {}
for col_idx in range(4, len(rows[header_row_idx])):
    m_name = str(rows[header_row_idx][col_idx]).strip()
    if m_name and m_name != 'None' and m_name != 'nan': 
        methods_found[col_idx] = m_name

blocks = []
current_block = None
for idx in range(header_row_idx + 1, len(rows)):
    row = rows[idx]
    if len(row) < 4: continue
    noise_raw = str(row[3]).strip().lower() if row[3] is not None else ''
    if 'clean' in noise_raw:
        current_block = {'dataset': None, 'rows': []}
        blocks.append(current_block)
    if current_block is not None:
        current_block['rows'].append(idx)
        ds_raw = str(row[2]).strip() if row[2] is not None else ''
        if ds_raw and ds_raw != 'None' and ds_raw != 'Dataset':
            ds_candidate = ds_raw.split()[-1].lower()
            current_block['dataset'] = EXCEL_DS_MAP.get(ds_candidate, ds_candidate)

for b in blocks:
    dataset = b['dataset']
    if not dataset: continue
    for idx in b['rows']:
        row = rows[idx]
        ntype, rate = parse_noise_label_excel(str(row[3]))
        if ntype is not None:
            for col_idx, m_name in methods_found.items():
                data = parse_val_with_std(str(row[col_idx]))
                add_result(m_name, dataset, ntype, rate, data)

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        dataset = row['dataset'].strip().lower()
        label = row['noise_rate'].strip().lower()
        if label.startswith('clean'): ntype, rate = 'clean', 0.0
        else:
            parts = label.split('_')
            ntype, rate = parts[0], float(parts[1])
        data = {'mean': float(row['lnpcc_mean']) * 100.0, 'std': float(row['lnpcc_std']) * 100.0}
        add_result('PCC+GCN', dataset, ntype, rate, data)

with open(INSTANCE_CSV_PATH, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        dataset = row['dataset'].strip().lower()
        rate = float(row['noise_rate'])
        method_raw = row['method'].strip().lower()
        if method_raw == 'lnpcc': method = 'PCC+GCN'
        elif method_raw == 'gcn': method = 'GCN'
        elif method_raw == 'cp': method = 'CP'
        elif method_raw == 'nrgnn': method = 'NRGNN'
        elif method_raw == 'pignn': method = 'PIGNN'
        else: method = method_raw.upper()
        data = {'mean': float(row['acc_mean']) * 100.0, 'std': float(row['acc_std']) * 100.0, 'cpu': float(row['cpu_s_avg']), 'gpu': float(row['gpu_s_avg'])}
        add_result(method, dataset, 'instance', rate, data)

DATASET_ORDER = ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', 'dblp', 'amazoncom', 'amazonpho', 'amazon-ratings', 'roman-empire']
instance_methods = ['GCN', 'CP', 'NRGNN', 'PIGNN', 'PCC+GCN']

print('\n' + '='*80)
print('TABLE 1: MEAN ACCURACY BY DATASET ON INSTANCE NOISE (Rates 0.1 - 0.5 average)')
print('='*80)
header = ['Method'] + [d.replace('-', ' ').title()[:9] for d in DATASET_ORDER] + ['AVERAGE']
print(f'{header[0]:<10}' + ''.join([f'{h:>10}' for h in header[1:]]))
print('-'*115)
for m in instance_methods:
    vals = []
    row_str = f'{m:<10}'
    for ds in DATASET_ORDER:
        ds_vals = [all_results[m][ds]['instance'][r]['mean'] for r in [0.1, 0.2, 0.3, 0.4, 0.5] if m in all_results and ds in all_results[m] and 'instance' in all_results[m][ds] and r in all_results[m][ds]['instance']]
        if ds_vals:
            avg_val = np.mean(ds_vals)
            vals.append(avg_val)
            row_str += f'{avg_val:>10.2f}'
        else:
            row_str += f'{"-":>10}'
    row_str += f'{np.mean(vals):>10.2f}' if vals else f'{"-":>10}'
    print(row_str)

print('\n' + '='*80)
print('TABLE 2: DETAILED TRAINING TIME BY DATASET (Instance Noise, Stacked CPU + GPU in seconds)')
print('='*80)
print(f'{"Method":<10}{"Metric":<8}' + ''.join([f'{d.replace("-", " ").title()[:9]:>10}' for d in DATASET_ORDER]) + f'{"AVERAGE":>10}')
print('-'*125)
for m in instance_methods:
    for metric, label in [('cpu', 'CPU'), ('gpu', 'GPU'), ('total', 'Total')]:
        row_str = f'{m:<10}{label:<8}'
        metric_vals = []
        for ds in DATASET_ORDER:
            times = []
            for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if m in all_results and ds in all_results[m] and 'instance' in all_results[m][ds] and r in all_results[m][ds]['instance']:
                    e = all_results[m][ds]['instance'][r]
                    if metric == 'cpu': times.append(e.get('cpu', 0))
                    elif metric == 'gpu': times.append(e.get('gpu', 0))
                    elif metric == 'total': times.append(e.get('cpu', 0) + e.get('gpu', 0))
            if times:
                avg_t = np.mean(times)
                metric_vals.append(avg_t)
                row_str += f'{avg_t:>10.2f}'
            else:
                row_str += f'{"-":>10}'
        row_str += f'{np.mean(metric_vals):>10.2f}' if metric_vals else f'{"-":>10}'
        print(row_str)
    print('-'*125)

print('\n' + '='*80)
print('TABLE 3: GLOBAL MEAN ACCURACY ACROSS ALL 10 DATASETS BY NOISE RATE')
print('='*80)
rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
for nt in ['instance', 'uniform', 'pair', 'random']:
    print(f'\n--- Noise Type: {nt.upper()} ---')
    methods_to_show = instance_methods if nt == 'instance' else sorted(all_results.keys())
    hdr = f'{"Method":<12}' + ''.join([f"r={r:<4.1f}" for r in rates]) + f'{"Avg(0.1-0.5)":>14}' + f'{"Overall":>10}'
    print(hdr)
    print('-'*70)
    for m in methods_to_show:
        r_means = {}
        for r in rates:
            curr_nt, curr_r = ('clean', 0.0) if r == 0.0 else (nt, r)
            ds_vals = [all_results[m][d][curr_nt][curr_r]['mean'] for d in DATASET_ORDER if m in all_results and d in all_results[m] and curr_nt in all_results[m][d] and curr_r in all_results[m][d][curr_nt]]
            if ds_vals:
                r_means[r] = np.mean(ds_vals)
        
        row_str = f'{m:<12}'
        for r in rates:
            if r in r_means: row_str += f'{r_means[r]:>8.2f}'
            else: row_str += f'{"-":>8}'
        
        noisy_avg = [r_means[r] for r in [0.1, 0.2, 0.3, 0.4, 0.5] if r in r_means]
        all_avg = [r_means[r] for r in rates if r in r_means]
        row_str += f'{(np.mean(noisy_avg) if noisy_avg else 0):>14.2f}'
        row_str += f'{(np.mean(all_avg) if all_avg else 0):>10.2f}'
        print(row_str)

print('\n' + '='*80)
print('TABLE 4: GLOBAL MEAN ACCURACY DELTA VS GCN (percentage points) BY NOISE RATE')
print('='*80)
for nt in ['instance', 'uniform', 'pair', 'random']:
    print(f'\n--- Noise Type: {nt.upper()} (Delta vs GCN) ---')
    methods_to_show = instance_methods if nt == 'instance' else sorted(all_results.keys())
    hdr = f'{"Method":<12}' + ''.join([f"r={r:<4.1f}" for r in rates]) + f'{"Avg Delta":>12}'
    print(hdr)
    print('-'*65)
    for m in methods_to_show:
        deltas = {}
        for r in rates:
            curr_nt, curr_r = ('clean', 0.0) if r == 0.0 else (nt, r)
            ds_deltas = []
            for d in DATASET_ORDER:
                if m in all_results and d in all_results[m] and curr_nt in all_results[m][d] and curr_r in all_results[m][d][curr_nt]:
                    if 'GCN' in all_results and d in all_results['GCN'] and curr_nt in all_results['GCN'][d] and curr_r in all_results['GCN'][d][curr_nt]:
                        delta = all_results[m][d][curr_nt][curr_r]['mean'] - all_results['GCN'][d][curr_nt][curr_r]['mean']
                        ds_deltas.append(delta)
            if ds_deltas:
                deltas[r] = np.mean(ds_deltas)
        
        row_str = f'{m:<12}'
        for r in rates:
            if r in deltas:
                v = deltas[r]
                sign = '+' if v > 0 else ''
                row_str += f'{sign}{v:>7.2f}'
            else:
                row_str += f'{"-":>8}'
        all_d = [deltas[r] for r in rates if r in deltas]
        if all_d:
            avg_d = np.mean(all_d)
            sign = '+' if avg_d > 0 else ''
            row_str += f'{sign:>4}{avg_d:>7.2f}'
        else:
            row_str += f'{"-":>12}'
        print(row_str)
