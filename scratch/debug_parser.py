import os
import re
import csv
import pandas as pd
from pathlib import Path

# Paths
XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
CSV_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\lnpcc_results_20260421_172348.csv'

EXCEL_DS_MAP = {
    'cora': 'cora',
    'citeseer': 'citeseer',
    'pubmed': 'pubmed',
    'amazon-c': 'amazoncom',
    'amazon-p': 'amazonpho',
    'dblp': 'dblp',
    'blogcatalog': 'blogcatalog',
    'flickr': 'flickr',
    'amz-rat.': 'amazon-ratings',
    'roman-emp.': 'roman-empire',
    'a-computers': 'amazoncom',
    'a-photo': 'amazonpho',
    'a-photos': 'amazonpho',
    'a-ratings': 'amazon-ratings',
    'empire': 'roman-empire',
    'amazonpho': 'amazonpho',
    'amazoncom': 'amazoncom',
}

def parse_noise_label_excel(noise_raw):
    noise_raw = str(noise_raw).strip().lower()
    if 'clean' in noise_raw:
        return 'clean', 0.0
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
    if nums: return float(nums[0])
    return None

all_results = {}

df_xls = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)

# Find header row
header_row_idx = -1
for i in range(15):
    if str(df_xls.iloc[i, 2]).strip() == "Dataset":
        header_row_idx = i
        break

methods_found = {}
for col_idx in range(4, len(df_xls.columns)):
    m_name = str(df_xls.iloc[header_row_idx, col_idx]).strip()
    if m_name and m_name != "nan":
        methods_found[col_idx] = m_name

# Block-based grouping (like generate_tables.py)
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
            # Extract last word and map
            ds_candidate = ds_raw.split()[-1].lower()
            current_block['dataset'] = EXCEL_DS_MAP.get(ds_candidate, ds_candidate)

# Process blocks
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

# Check for scores > 1
print("Summary of Clean Accuracies:")
for method, datasets in all_results.items():
    print(f"\nMethod: {method}")
    for ds, ntypes in datasets.items():
        if 'clean' in ntypes:
            clean = ntypes['clean'][0.0]
            max_acc = 0
            for nt, rates in ntypes.items():
                for r, acc in rates.items():
                    if acc > max_acc: max_acc = acc
            
            status = "OK" if max_acc <= clean + 1e-5 else "!! ERROR: NOISE > CLEAN !!"
            print(f"  {ds:15}: Clean={clean:.2f}, Max={max_acc:.2f} -> {status}")
