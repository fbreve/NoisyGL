import sys
print("Init...", flush=True)
import os
import re
import csv
print("Basic imports done...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("Matplotlib done...", flush=True)
import numpy as np
print("Numpy done...", flush=True)
import openpyxl

# Paths
XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
CSV_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\lnpcc_results_20260421_172348.csv'
INSTANCE_CSV_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\instance_final_results.csv'
BASE_OUTPUT_DIR = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results'

# Mappings
EXCEL_DS_MAP = {
    'cora': 'cora', 'citeseer': 'citeseer', 'pubmed': 'pubmed',
    'amazon-c': 'amazoncom', 'amazon-p': 'amazonpho', 'dblp': 'dblp',
    'blogcatalog': 'blogcatalog', 'flickr': 'flickr', 'amz-rat.': 'amazon-ratings',
    'roman-emp.': 'roman-empire', 'a-computers': 'amazoncom', 'a-photo': 'amazonpho',
    'a-photos': 'amazonpho', 'a-ratings': 'amazon-ratings', 'empire': 'roman-empire',
    'amazonpho': 'amazonpho', 'amazoncom': 'amazoncom',
}

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
    if len(nums) >= 2:
        return {'mean': float(nums[0]), 'std': float(nums[1])}
    elif len(nums) == 1:
        return {'mean': float(nums[0]), 'std': 0.0}
    return None

all_results = {} # all_results[method][dataset][ntype][rate] = {'mean': m, 'std': s}
def add_result(method, dataset, ntype, rate, data):
    if data is None: return
    if method not in all_results: all_results[method] = {}
    if dataset not in all_results[method]: all_results[method][dataset] = {}
    if ntype not in all_results[method][dataset]: all_results[method][dataset][ntype] = {}
    all_results[method][dataset][ntype][rate] = data

# 1. Parse Excel using openpyxl directly (more stable on this environment)
print(f"Reading Excel from {XLSX_PATH}...", flush=True)
import openpyxl
wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
sheet = wb.active
print(f"Excel loaded. Sheet: {sheet.title}", flush=True)

header_row_idx = -1
rows = list(sheet.iter_rows(values_only=True))

for i in range(20):
    if len(rows[i]) > 2 and str(rows[i][2]).strip() == "Dataset":
        header_row_idx = i
        break

methods_found = {}
for col_idx in range(4, len(rows[header_row_idx])):
    m_name = str(rows[header_row_idx][col_idx]).strip()
    if m_name and m_name != "None" and m_name != "nan": 
        methods_found[col_idx] = m_name

blocks = []
current_block = None
for idx in range(header_row_idx + 1, len(rows)):
    row = rows[idx]
    if len(row) < 4: continue
    noise_raw = str(row[3]).strip().lower() if row[3] is not None else ""
    if 'clean' in noise_raw:
        current_block = {'dataset': None, 'rows': []}
        blocks.append(current_block)
    if current_block is not None:
        current_block['rows'].append(idx)
        ds_raw = str(row[2]).strip() if row[2] is not None else ""
        if ds_raw and ds_raw != "None" and ds_raw != "Dataset":
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
print("Excel parsing complete.", flush=True)

# 2. Parse LNPCC CSV for CLEAN only
print(f"Reading LNPCC CSV from {CSV_PATH} for Clean data...")
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['dataset'].strip().lower()
        label = row['noise_rate'].strip().lower()
        if label.startswith('clean'): ntype, rate = 'clean', 0.0
        else:
            parts = label.split('_')
            ntype, rate = parts[0], float(parts[1])
        data = {
            'mean': float(row['lnpcc_mean']) * 100.0,
            'std': float(row['lnpcc_std']) * 100.0
        }
        add_result('LN-PCC', dataset, ntype, rate, data)

# 3. Parse Instance CSV
print(f"Reading Instance CSV from {INSTANCE_CSV_PATH}...")
with open(INSTANCE_CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['dataset'].strip().lower()
        rate = float(row['noise_rate'])
        method_raw = row['method'].strip().lower()
        # map method names to match excel
        if method_raw == 'lnpcc': method = 'LN-PCC'
        elif method_raw == 'gcn': method = 'GCN'
        elif method_raw == 'cp': method = 'CP'
        elif method_raw == 'nrgnn': method = 'NRGNN'
        elif method_raw == 'pignn': method = 'PIGNN'
        else: method = method_raw.upper()
        
        data = {
            'mean': float(row['acc_mean']) * 100.0,
            'std': float(row['acc_std']) * 100.0,
            'cpu': float(row['cpu_s_avg']),
            'gpu': float(row['gpu_s_avg'])
        }
        add_result(method, dataset, 'instance', rate, data)

noise_types = ['instance', 'uniform', 'pair', 'random']
DATASET_ORDER = ['cora', 'citeseer', 'pubmed', 'blogcatalog', 'flickr', 'dblp', 'amazoncom', 'amazonpho', 'amazon-ratings', 'roman-empire']
all_datasets = [ds for ds in DATASET_ORDER if any(ds in all_results[m] for m in all_results)]

# We only care about methods that are present in the instance experiments
instance_methods = ['GCN', 'CP', 'NRGNN', 'PIGNN', 'LN-PCC']

def find_best_baseline_for_dataset(ds, mode='accuracy'):
    winner, max_avg = None, -999.0
    for m in instance_methods:
        if m in ['LN-PCC', 'GCN'] or m not in all_results or ds not in all_results[m]: continue
        vals = []
        if 'instance' in all_results[m][ds]:
            for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                if r == 0.0 and 'clean' in all_results[m][ds] and 0.0 in all_results[m][ds]['clean']:
                    val = all_results[m][ds]['clean'][0.0]['mean']
                elif r in all_results[m][ds]['instance']:
                    val = all_results[m][ds]['instance'][r]['mean']
                else:
                    continue
                
                if mode == 'delta':
                    if r == 0.0 and 'clean' in all_results['GCN'][ds] and 0.0 in all_results['GCN'][ds]['clean']:
                        gcn_val = all_results['GCN'][ds]['clean'][0.0]['mean']
                    elif r in all_results['GCN'][ds]['instance']:
                        gcn_val = all_results['GCN'][ds]['instance'][r]['mean']
                    else:
                        continue
                    val -= gcn_val
                vals.append(val)
        if vals:
            avg = np.mean(vals)
            if avg > max_avg: max_avg, winner = avg, m
    return winner

def find_global_best_complete(mode='accuracy'):
    winner, max_avg = None, -999.0
    for m in instance_methods:
        if m in ['LN-PCC', 'GCN'] or m not in all_results: continue
        vals = []
        for ds in all_datasets:
            for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                if r == 0.0:
                    curr_nt, curr_r = 'clean', 0.0
                else:
                    curr_nt, curr_r = 'instance', r
                    
                if ds in all_results[m] and curr_nt in all_results[m][ds] and curr_r in all_results[m][ds][curr_nt]:
                    val = all_results[m][ds][curr_nt][curr_r]['mean']
                    if mode == 'delta':
                        if curr_nt in all_results['GCN'][ds] and curr_r in all_results['GCN'][ds][curr_nt]:
                            val -= all_results['GCN'][ds][curr_nt][curr_r]['mean']
                        else:
                            continue
                    vals.append(val)
        if vals:
            avg = np.mean(vals)
            if avg > max_avg: max_avg, winner = avg, m
    return winner

plt.style.use('ggplot')
# Dynamic Color Mapping for all methods
all_found_methods = sorted(all_results.keys())
cmap = plt.cm.get_cmap('tab20', len(all_found_methods))
method_colors = {m: cmap(i) for i, m in enumerate(all_found_methods)}
# Keep specific brand colors for core methods
core_overrides = {
    'GCN': '#1f77b4', 
    'CP': '#ff7f0e', 
    'NRGNN': '#2ca02c', 
    'PIGNN': '#9467bd', 
    'LN-PCC': '#d62728',
    'CGNN': '#8c564b'  # Distinct Brown for CGNN
}
for m, c in core_overrides.items():
    if m in method_colors: method_colors[m] = c

def plot_lines(ax, ds, nt, best_baseline, methods_to_show, mode='accuracy'):
    # Highlight logic
    prio = ['LN-PCC', 'GCN']
    if best_baseline and best_baseline not in prio:
        prio.append(best_baseline)
        
    # Sort for legend: GCN first, others in between, LN-PCC last
    others = [m for m in methods_to_show if m not in ['GCN', 'LN-PCC']]
    # We sort 'others' alphabetically to keep the middle part organized
    others.sort()
    sorted_methods = (['GCN'] if 'GCN' in methods_to_show else []) + \
                     others + \
                     (['LN-PCC'] if 'LN-PCC' in methods_to_show else [])
    
    has_data = False
    for m in sorted_methods:
        plot_rates, plot_means, plot_stds = [], [], []
        for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            curr_nt, curr_r = ('clean', 0.0) if r == 0.0 else (nt, r)
            vals = []
            for d in (all_datasets if ds is None else [ds]):
                if m in all_results and d in all_results[m] and curr_nt in all_results[m][d] and curr_r in all_results[m][d][curr_nt]:
                    val = all_results[m][d][curr_nt][curr_r]['mean']
                    if mode == 'delta':
                        if 'GCN' in all_results and d in all_results['GCN'] and curr_nt in all_results['GCN'][d] and curr_r in all_results['GCN'][d][curr_nt]:
                            val -= all_results['GCN'][d][curr_nt][curr_r]['mean']
                        else: continue
                    vals.append(val)
            if vals:
                plot_rates.append(r)
                plot_means.append(np.mean(vals))
                if ds is None: plot_stds.append(np.std(vals))
                else: plot_stds.append(all_results[m][ds][curr_nt][curr_r]['std'])

        if plot_means:
            has_data = True
            color = method_colors.get(m, 'gray')
            if m in prio:
                lw = 2.6
                alpha = 1.0
                zorder = 20 if m == 'LN-PCC' else 15
            else:
                lw = 1.2
                alpha = 0.5
                zorder = 5
            
            ax.errorbar(plot_rates, plot_means, yerr=plot_stds, label=m, color=color, 
                        linewidth=lw, alpha=alpha, marker='o', markersize=4, 
                        capsize=3, elinewidth=0.8, capthick=0.8, zorder=zorder)
    
    ax.set_xlabel("Noise Rate", fontsize=11)
    ax.set_ylabel("Accuracy (%)" if mode == 'accuracy' else "Delta vs GCN (%)", fontsize=11)
    return has_data

def save_dual(fig, base_path, dpi=300):
    fig.savefig(base_path + ".png", dpi=dpi, bbox_inches='tight')
    fig.savefig(base_path + ".pdf", bbox_inches='tight')

def plot_and_save(ds, nt, mode, local_best, global_best, ds_mode):
    # Rename to detailed instead of instance
    subdir = 'plots_detailed_accuracy' if mode == 'accuracy' else 'plots_detailed_delta'
    output_dir = os.path.join(BASE_OUTPUT_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    methods_to_plot = instance_methods if nt == 'instance' else sorted(all_results.keys())
    if plot_lines(ax, ds, nt, local_best, methods_to_plot, mode):
        # Force lowercase filename
        fname = (ds if ds else "average").lower()
        title_name = ds.capitalize() if ds else "Average"
        ax.set_title(f"{mode.capitalize()}: {title_name} - {nt.capitalize()} Noise", fontsize=15, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        save_dual(fig, os.path.join(output_dir, f"{fname}_{nt}"))
    plt.close()

def run_visualization_mode(mode='accuracy', skip_individual=False):
    global_best = find_global_best_complete(mode)
    if not skip_individual:
        for ds in all_datasets:
            local_best = find_best_baseline_for_dataset(ds, mode)
            for nt in noise_types: plot_and_save(ds, nt, mode, local_best, global_best, True)
    for nt in noise_types: plot_and_save(None, nt, mode, None, global_best, False)

def plot_bar_chart_average_accuracy(ntype='instance'):
    data_records = []
    rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    methods_to_use = instance_methods if ntype == 'instance' else sorted(all_results.keys())
    method_totals = {m: [] for m in methods_to_use}
    
    for ds in all_datasets:
        for m in methods_to_use:
            vals = [all_results[m][ds][ntype][r]['mean'] for r in rates if m in all_results and ds in all_results[m] and ntype in all_results[m][ds] and r in all_results[m][ds][ntype]]
            if vals:
                acc = sum(vals)/len(vals)
                # Save with raw dataset name for matching
                data_records.append({'ds_raw': ds, 'Method': m, 'Accuracy': acc})
                method_totals[m].append(acc)
    
    for m in methods_to_use:
        if method_totals[m]: 
            data_records.append({'ds_raw': 'average', 'Method': m, 'Accuracy': sum(method_totals[m])/len(method_totals[m])})
            
    if not data_records: return
    
    # Define order using raw names
    active_ds = list(set(r['ds_raw'] for r in data_records if r['ds_raw'] != 'average'))
    ds_order_raw = [d for d in DATASET_ORDER if d in active_ds] + ['average']
    # Display names for labels
    ds_labels = [d.replace('-', ' ').replace('_', ' ').title() if d != 'average' else 'AVERAGE' for d in ds_order_raw]
    
    # Force order: GCN first, LN-PCC last
    others = [m for m in methods_to_use if m not in ['GCN', 'LN-PCC']]
    methods_to_use = (['GCN'] if 'GCN' in methods_to_use else []) + others + (['LN-PCC'] if 'LN-PCC' in methods_to_use else [])
    
    fig, ax = plt.subplots(figsize=(18, 8))
    indices = np.arange(len(ds_order_raw))
    bw = 0.8 / len(methods_to_use)
    
    for i, m in enumerate(methods_to_use):
        accs = []
        for d in ds_order_raw:
            val = next((r['Accuracy'] for r in data_records if r['ds_raw'] == d and r['Method'] == m), 0)
            accs.append(val)
        ax.bar(indices + (i - len(methods_to_use)/2 + 0.5)*bw, accs, bw, label=m, color=method_colors.get(m, 'gray'), edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'Mean Accuracy Comparison: {ntype.upper()} Noise', fontsize=18, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels(ds_labels, rotation=45, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=14); ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    save_dual(fig, os.path.join(BASE_OUTPUT_DIR, 'plots_bar_average', f"bar_avg_{ntype}"))
    plt.close()

def plot_bar_chart_stacked_times():
    data_records = []
    method_cpu_totals = {m: [] for m in instance_methods}
    method_gpu_totals = {m: [] for m in instance_methods}
    
    for ds in all_datasets:
        for m in instance_methods:
            cpu, gpu = [], []
            for r in [0.1,0.2,0.3,0.4,0.5]:
                if m in all_results and ds in all_results[m] and 'instance' in all_results[m][ds] and r in all_results[m][ds]['instance']:
                    entry = all_results[m][ds]['instance'][r]
                    if 'cpu' in entry: cpu.append(entry['cpu']); gpu.append(entry['gpu'])
            if cpu: 
                ac, ag = sum(cpu)/len(cpu), sum(gpu)/len(gpu)
                data_records.append({'ds_raw': ds, 'Method': m, 'CPU': ac, 'GPU': ag})
                method_cpu_totals[m].append(ac); method_gpu_totals[m].append(ag)
                
    for m in instance_methods:
        if method_cpu_totals[m]:
            data_records.append({'ds_raw': 'average', 'Method': m, 'CPU': sum(method_cpu_totals[m])/len(method_cpu_totals[m]), 'GPU': sum(method_gpu_totals[m])/len(method_gpu_totals[m])})
            
    if not data_records: return
    active_ds = list(set(r['ds_raw'] for r in data_records if r['ds_raw'] != 'average'))
    ds_order_raw = [d for d in DATASET_ORDER if d in active_ds] + ['average']
    ds_labels = [d.replace('-', ' ').replace('_', ' ').title() if d != 'average' else 'AVERAGE' for d in ds_order_raw]
    
    # Force order: GCN first, LN-PCC last
    others = [m for m in instance_methods if m not in ['GCN', 'LN-PCC']]
    methods_ordered = (['GCN'] if 'GCN' in instance_methods else []) + others + (['LN-PCC'] if 'LN-PCC' in instance_methods else [])
    
    fig, ax = plt.subplots(figsize=(20, 9))
    bw = 0.8 / len(methods_ordered); indices = np.arange(len(ds_order_raw))
    
    for i, m in enumerate(methods_ordered):
        cv = [next((r['CPU'] for r in data_records if r['ds_raw'] == d and r['Method'] == m), 0) for d in ds_order_raw]
        gv = [next((r['GPU'] for r in data_records if r['ds_raw'] == d and r['Method'] == m), 0) for d in ds_order_raw]
        x_pos = indices + (i - len(instance_methods)/2 + 0.5)*bw
        ax.bar(x_pos, cv, bw, color=method_colors.get(m,'gray'), alpha=0.4, edgecolor='black')
        ax.bar(x_pos, gv, bw, bottom=cv, color=method_colors.get(m,'gray'), alpha=0.9, edgecolor='black')
        
    ax.set_title("Mean Training Time Comparison (Stacked CPU+GPU)", fontsize=18, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels(ds_labels, rotation=45, fontsize=11); ax.set_ylabel("Time (seconds)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=method_colors[m], lw=4, label=m) for m in methods_ordered]
    legend_elements.extend([Line2D([0], [0], color='gray', alpha=0.4, lw=4, label='CPU Time'), Line2D([0], [0], color='gray', alpha=0.9, lw=4, label='GPU Time')])
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    plt.tight_layout(); save_dual(fig, os.path.join(BASE_OUTPUT_DIR, 'plots_instance_bar', "stacked_times")); plt.close()

def plot_summary_classic_noises(mode='accuracy'):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    methods = sorted(all_results.keys())
    for i, nt in enumerate(['uniform', 'pair', 'random']):
        plot_lines(axes[i], None, nt, find_global_best_complete(mode), methods, mode)
        axes[i].set_title(nt.capitalize() + " Noise", fontsize=17, fontweight='bold')
        if i == 2: axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.suptitle(f"Global Benchmark Summary ({mode.capitalize()})", fontsize=20, y=1.02)
    plt.tight_layout(); save_dual(fig, os.path.join(BASE_OUTPUT_DIR, 'summary_plots', f"summary_classic_{mode}")); plt.close()

def plot_summary_instance_noise(mode='accuracy'):
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_lines(ax, None, 'instance', find_global_best_complete(mode), instance_methods, mode)
    ax.set_title(f"Instance Noise Global Summary ({mode.capitalize()})", fontsize=17, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.tight_layout(); save_dual(fig, os.path.join(BASE_OUTPUT_DIR, 'summary_plots', f"summary_instance_{mode}")); plt.close()

def plot_dataset_grid(ntype='instance', mode='accuracy'):
    fig, axes = plt.subplots(2, 5, figsize=(25, 13))
    methods = instance_methods if ntype == 'instance' else sorted(all_results.keys())
    for i, ds in enumerate(all_datasets):
        ax = axes.flatten()[i]
        plot_lines(ax, ds, ntype, find_best_baseline_for_dataset(ds, mode), methods, mode)
        # Cleaner dataset titles
        ax.set_title(ds.replace('-', ' ').replace('_', ' ').title(), fontsize=16, fontweight='bold')
    handles, labels = axes[0,0].get_legend_handles_labels()
    # Force ncol=6 for symmetric 2-row legend if 12 methods
    n_col = 6 if len(methods) >= 6 else len(methods)
    fig.legend(handles, labels, loc='lower center', ncol=n_col, fontsize=13, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    save_dual(fig, os.path.join(BASE_OUTPUT_DIR, 'dataset_grids', f"grid_{ntype}_{mode}")); plt.close()

SKIP_INDIVIDUAL = True
for d in ['summary_plots', 'dataset_grids', 'plots_bar_average', 'plots_instance_bar', 'plots_detailed_accuracy', 'plots_detailed_delta']:
    os.makedirs(os.path.join(BASE_OUTPUT_DIR, d), exist_ok=True)

for mode in ['accuracy', 'delta']:
    run_visualization_mode(mode, SKIP_INDIVIDUAL)
    plot_summary_classic_noises(mode)
    plot_summary_instance_noise(mode)
    for nt in noise_types: plot_dataset_grid(nt, mode)

for nt in noise_types: plot_bar_chart_average_accuracy(nt)
plot_bar_chart_stacked_times()
print(f"Completed! Plots generated in {BASE_OUTPUT_DIR}")
