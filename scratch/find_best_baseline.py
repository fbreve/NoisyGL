import os
import re
import csv
import pandas as pd
from pathlib import Path

# Paths
XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
EXCEL_DS_MAP = {
    'cora': 'cora', 'citeseer': 'citeseer', 'pubmed': 'pubmed',
    'amazon-c': 'amazoncom', 'amazon-p': 'amazonpho', 'dblp': 'dblp',
    'blogcatalog': 'blogcatalog', 'flickr': 'flickr', 'amz-rat.': 'amazon-ratings',
    'roman-emp.': 'roman-empire', 'a-computers': 'amazoncom', 'a-photo': 'amazonpho',
    'a-photos': 'amazonpho', 'a-ratings': 'amazon-ratings', 'empire': 'roman-empire'
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

def parse_val(val_str):
    if not val_str or str(val_str).lower() == 'nan': return None
    nums = re.findall(r'[\d.]+', str(val_str))
    return float(nums[0]) if nums else None

df_xls = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)
header_row_idx = -1
for i in range(15):
    if str(df_xls.iloc[i, 2]).strip() == "Dataset":
        header_row_idx = i
        break

methods_found = {}
for col_idx in range(4, len(df_xls.columns)):
    m_name = str(df_xls.iloc[header_row_idx, col_idx]).strip()
    if m_name and m_name != "nan": methods_found[col_idx] = m_name

# Block-based identification
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

all_results = {}
for b in blocks:
    dataset = b['dataset']
    if not dataset: continue
    for idx in b['rows']:
        row = df_xls.iloc[idx]
        ntype, rate = parse_noise_label_excel(row[3])
        if ntype is None: continue
        for col_idx, m_name in methods_found.items():
            acc = parse_val(row[col_idx])
            if acc is not None:
                if m_name not in all_results: all_results[m_name] = {}
                if dataset not in all_results[m_name]: all_results[m_name][dataset] = {}
                if ntype not in all_results[m_name][dataset]: all_results[m_name][dataset][ntype] = {}
                all_results[m_name][dataset][ntype][rate] = acc

# Calculate global mean score per method
method_avg_score = {}
for m, datasets in all_results.items():
    if m.lower() == 'gcn': continue # Exclude GCN from "Best baseline"
    scores = []
    for ds, ntypes in datasets.items():
        if 'clean' in ntypes:
            clean = ntypes['clean'][0.0]
            if clean == 0: continue
            for nt, rates in ntypes.items():
                if nt == 'clean': continue
                for r, acc in rates.items():
                    scores.append(acc / clean)
    if scores:
        method_avg_score[m] = sum(scores) / len(scores)

# Rank them
sorted_methods = sorted(method_avg_score.items(), key=lambda x: x[1], reverse=True)
print("Method Ranking by Mean Score:")
for m, score in sorted_methods:
    print(f"  {m:15}: {score:.4f}")

if sorted_methods:
    print(f"\nBEST_METHOD = '{sorted_methods[0][0]}'")
