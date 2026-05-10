import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Paths
XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
CSV_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\lnpcc_results_20260421_172348.csv'
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

# 1. Parse Excel
print(f"Reading Excel from {XLSX_PATH}...")
df_xls = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)
header_row_idx = -1
for i in range(15):
    if str(df_xls.iloc[i, 2]).strip() == "Dataset":
        header_row_idx = i
        break

if header_row_idx == -1: exit(1)

methods_found = {}
for col_idx in range(4, len(df_xls.columns)):
    m_name = str(df_xls.iloc[header_row_idx, col_idx]).strip()
    if m_name and m_name != "nan": methods_found[col_idx] = m_name

blocks = []
current_block = None
for idx in range(header_row_idx + 1, len(df_xls)):
    row = df_xls.iloc[idx]
    noise_raw = str(row[3]).strip().lower() if pd.notna(row[3]) else ""
    if 'clean' in noise_raw:
        current_block = {'dataset': None, 'rows': []}
        blocks.append(current_block)
    if current_block is not None:
        current_block['rows'].append(idx)
        ds_raw = str(row[2]).strip() if pd.notna(row[2]) else ""
        if ds_raw and ds_raw != "nan" and ds_raw != "Dataset":
            ds_candidate = ds_raw.split()[-1].lower()
            current_block['dataset'] = EXCEL_DS_MAP.get(ds_candidate, ds_candidate)

for b in blocks:
    dataset = b['dataset']
    if not dataset: continue
    for idx in b['rows']:
        row = df_xls.iloc[idx]
        ntype, rate = parse_noise_label_excel(row[3])
        if ntype is None: continue
        for col_idx, m_name in methods_found.items():
            data = parse_val_with_std(row[col_idx])
            add_result(m_name, dataset, ntype, rate, data)

# 2. Parse LNPCC CSV
print(f"Reading LNPCC CSV from {CSV_PATH}...")
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

# 3. Helpers
noise_types = ['uniform', 'pair', 'random']
all_datasets = sorted(list(set(ds for m in all_results for ds in all_results[m])))
all_methods = sorted(list(all_results.keys()))
complete_methods = [m for m in all_methods if len(all_results[m]) == len(all_datasets)]

def find_best_baseline_for_dataset(ds, mode='accuracy'):
    winner, max_avg = None, -999.0
    for m in all_methods:
        if m in ['LN-PCC', 'GCN'] or ds not in all_results[m]: continue
        vals = []
        for nt in noise_types:
            if nt in all_results[m][ds]:
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    if r in all_results[m][ds][nt] or (r == 0.0 and 'clean' in all_results[m][ds] and 0.0 in all_results[m][ds]['clean']):
                        val = all_results[m][ds]['clean'][0.0]['mean'] if r == 0.0 else all_results[m][ds][nt][r]['mean']
                        if mode == 'delta':
                            gcn_val = all_results['GCN'][ds]['clean'][0.0]['mean'] if r == 0.0 else all_results['GCN'][ds][nt][r]['mean']
                            val -= gcn_val
                        vals.append(val)
        if vals:
            avg = np.mean(vals)
            if avg > max_avg: max_avg, winner = avg, m
    return winner

def find_global_best_complete(mode='accuracy'):
    winner, max_avg = None, -999.0
    for m in complete_methods:
        if m in ['LN-PCC', 'GCN']: continue
        vals = []
        for ds in all_datasets:
            for nt in noise_types:
                for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    curr_nt, curr_r = ('clean', 0.0) if r == 0.0 else (nt, r)
                    if m in all_results and ds in all_results[m] and curr_nt in all_results[m][ds] and curr_r in all_results[m][ds][curr_nt]:
                        val = all_results[m][ds][curr_nt][curr_r]['mean']
                        if mode == 'delta':
                            val -= all_results['GCN'][ds][curr_nt][curr_r]['mean']
                        vals.append(val)
        if vals:
            avg = np.mean(vals)
            if avg > max_avg: max_avg, winner = avg, m
    return winner

# 4. Plotting
sns.set_theme(style="whitegrid")
palette = sns.color_palette("muted", len(all_methods))
method_colors = {m: palette[i] for i, m in enumerate(all_methods)}
method_colors['LN-PCC'] = 'red'
method_colors['GCN'] = 'blue'

def plot_lines(ax, ds, nt, best_baseline, methods_to_show, mode='accuracy'):
    prio = ['LN-PCC', 'GCN']
    if best_baseline: prio.append(best_baseline)
    
    draw_order = [m for m in methods_to_show if m not in prio] + [m for m in prio]
    has_data = False
    
    for m in draw_order:
        rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        plot_rates, plot_means, plot_stds = [], [], []
        
        for r in rates:
            curr_nt, curr_r = ('clean', 0.0) if r == 0.0 else (nt, r)
            if ds: # Individual
                if m in all_results and ds in all_results[m] and curr_nt in all_results[m][ds] and curr_r in all_results[m][ds][curr_nt]:
                    mean = all_results[m][ds][curr_nt][curr_r]['mean']
                    std = all_results[m][ds][curr_nt][curr_r]['std']
                    if mode == 'delta':
                        gcn_data = all_results['GCN'][ds][curr_nt][curr_r]
                        mean -= gcn_data['mean']
                        # Std of delta is complex, user asked for visualization of stability, we'll use original std for shadow
                    plot_rates.append(r)
                    plot_means.append(mean)
                    plot_stds.append(std)
            else: # Aggregate
                vals = []
                for d in all_datasets:
                    if m in all_results and d in all_results[m] and curr_nt in all_results[m][d] and curr_r in all_results[m][d][curr_nt]:
                        val = all_results[m][d][curr_nt][curr_r]['mean']
                        if mode == 'delta':
                            val -= all_results['GCN'][d][curr_nt][curr_r]['mean']
                        vals.append(val)
                if vals:
                    plot_rates.append(r)
                    plot_means.append(np.mean(vals))
                    plot_stds.append(np.std(vals))

        if plot_means:
            color = 'green' if m == best_baseline else method_colors.get(m, 'gray')
            if m == 'LN-PCC': lw, alpha, shaded, shade_alpha, zorder = 4.5, 1.0, True, 0.2, 10
            elif m == 'GCN': lw, alpha, shaded, shade_alpha, zorder = 2.5, 0.9, False, 0.0, 8
            elif m == best_baseline: lw, alpha, shaded, shade_alpha, zorder = 2.5, 0.9, True, 0.1, 8
            else: lw, alpha, shaded, shade_alpha, zorder = 0.8, 0.4, False, 0.0, 2
            
            ax.plot(plot_rates, plot_means, marker='o' if lw > 1 else None, 
                    label=m, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            if shaded:
                means_arr, stds_arr = np.array(plot_means), np.array(plot_stds)
                ax.fill_between(plot_rates, means_arr - stds_arr, means_arr + stds_arr, 
                                color=color, alpha=shade_alpha, zorder=zorder-1)
            has_data = True
    return has_data

def run_visualization_mode(mode='accuracy'):
    subdir = 'plots_accuracy' if mode == 'accuracy' else 'plots_delta'
    output_dir = os.path.join(BASE_OUTPUT_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    global_best = find_global_best_complete(mode)
    print(f"[{mode.upper()}] Global best baseline: {global_best}")

    for ds in all_datasets:
        local_best = find_best_baseline_for_dataset(ds, mode)
        for nt in noise_types:
            fig, ax = plt.subplots(figsize=(10, 7))
            if plot_lines(ax, ds, nt, local_best, all_methods, mode):
                title_prefix = "Accuracy Curve" if mode == 'accuracy' else "Accuracy Delta vs GCN"
                y_label = "Accuracy (%)" if mode == 'accuracy' else "Accuracy Delta (%)"
                ax.set_title(f"{title_prefix}: {ds.capitalize()} - {nt.capitalize()} Noise", fontsize=15)
                ax.set_xlabel("Noise Rate", fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{ds}_{nt}.png"), dpi=150)
            plt.close()

    for nt in noise_types:
        fig, ax = plt.subplots(figsize=(10, 7))
        if plot_lines(ax, None, nt, global_best, complete_methods, mode):
            title_prefix = "Mean Accuracy Curve" if mode == 'accuracy' else "Mean Accuracy Delta vs GCN"
            y_label = "Mean Accuracy (%)" if mode == 'accuracy' else "Mean Accuracy Delta (%)"
            ax.set_title(f"{title_prefix} - {nt.capitalize()} Noise", fontsize=15)
            ax.set_xlabel("Noise Rate", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"average_{nt}.png"), dpi=150)
        plt.close()

run_visualization_mode('accuracy')
run_visualization_mode('delta')
print(f"Completed! Plots generated in {BASE_OUTPUT_DIR}")
