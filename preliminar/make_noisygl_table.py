"""
make_noisygl_table.py
Generates LaTeX tables in NoisyGL paper style (Tables A3/A4) from the CSV
produced by total_exp_lnpcc.py.

Table structure (one table per noise_type):
  Col 1 : Noise Type  (multirow, e.g. "Uniform")
  Col 2 : Noise Rate  (0.1, 0.2, ..., 0.5  or  "Clean" for rate 0.0)
  Col 3+: One column per dataset (Cora, CiteSeer, PubMed, ...)
  Values: mean ± std (accuracy in %)

  One row per (noise_type, noise_rate), one sub-table per variant/method.
  Bold = best value in each dataset column.

Usage:
  # Single CSV
  python make_noisygl_table.py lnpcc_results_20260312_120000.csv

  # With NoisyGL paper results CSV for comparison
  python make_noisygl_table.py lnpcc_results_*.csv --noisygl_csv noisygl_paper_results.csv

  # Select specific variants and datasets
  python make_noisygl_table.py results.csv \\
      --variants rem_od rel_od rem_s_od rel_s_od \\
      --datasets cora citeseer pubmed

  # Output to file
  python make_noisygl_table.py results.csv --output tables.tex

NoisyGL paper results CSV format (--noisygl_csv):
  A CSV with columns: method, dataset, noise_type, noise_rate, mean, std
  where method is the NoisyGL method name (e.g. 'gcn', 'nrgnn', etc.)
  and mean/std are already in percentage (0-100).
  This CSV must be built manually from the paper's tables.
"""

import argparse
import sys
import os
import csv
import re
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None

# ── Config ────────────────────────────────────────────────────────────────────

NOISE_TYPE_LABELS = {
    'clean':    'Clean',
    'uniform':  'Uniform',
    'pair':     'Pair',
    'random':   'Random',
    'instance': 'Instance',
}

NOISE_TYPE_ORDER = ['clean', 'uniform', 'pair', 'random', 'instance']

DATASET_LABELS = {
    'cora':          'Cora',
    'citeseer':      'CiteSeer',
    'pubmed':        'PubMed',
    'amazoncom':     'Amazon-C',
    'amazonpho':     'Amazon-P',
    'dblp':          'DBLP',
    'blogcatalog':   'BlogCatalog',
    'flickr':        'Flickr',
    'amazon-ratings':'Amz-Rat.',
    'roman-empire':  'Roman-Emp.',
}

VARIANT_LABELS = {
    'rem_def':   r'LN-PCC$_{\text{rem}}^{\text{def}}$',
    'rel_def':   r'LN-PCC$_{\text{rel}}^{\text{def}}$',
    'rem_od':    r'LN-PCC$_{\text{rem}}^{\text{od}}$',
    'rel_od':    r'LN-PCC$_{\text{rel}}^{\text{od}}$',
    'rem_s_def': r'LN-PCC$_{\text{rem,S}}^{\text{def}}$',
    'rel_s_def': r'LN-PCC$_{\text{rel,S}}^{\text{def}}$',
    'rem_s_od':  r'LN-PCC$_{\text{rem,S}}^{\text{od}}$',
    'rel_s_od':  r'LN-PCC$_{\text{rel,S}}^{\text{od}}$',
    'rem_d_def': r'LN-PCC$_{\text{rem,D}}^{\text{def}}$',
    'rel_d_def': r'LN-PCC$_{\text{rel,D}}^{\text{def}}$',
    'rem_d_od':  r'LN-PCC$_{\text{rem,D}}^{\text{od}}$',
    'rel_d_od':  r'LN-PCC$_{\text{rel,D}}^{\text{od}}$',
    'rem_p_def': r'LN-PCC$_{\text{rem,P}}^{\text{def}}$',
    'rel_p_def': r'LN-PCC$_{\text{rel,P}}^{\text{def}}$',
    'rem_p_od':  r'LN-PCC$_{\text{rem,P}}^{\text{od}}$',
    'rel_p_od':  r'LN-PCC$_{\text{rel,P}}^{\text{od}}$',
    'rem_od_nni':   r'LN-PCC$_{\text{rem}}^{\text{od-nni}}$',
    'rel_od_nni':   r'LN-PCC$_{\text{rel}}^{\text{od-nni}}$',
    'rem_s_od_nni': r'LN-PCC$_{\text{rem,S}}^{\text{od-nni}}$',
    'rel_s_od_nni': r'LN-PCC$_{\text{rel,S}}^{\text{od-nni}}$',
    'rem_d_od_nni': r'LN-PCC$_{\text{rem,D}}^{\text{od-nni}}$',
    'rel_d_od_nni': r'LN-PCC$_{\text{rel,D}}^{\text{od-nni}}$',
    'rem_p_od_nni': r'LN-PCC$_{\text{rem,P}}^{\text{od-nni}}$',
    'rel_p_od_nni': r'LN-PCC$_{\text{rel,P}}^{\text{od-nni}}$',
    # Baselines
    'gcn':           r'GCN',
    'pcc_def':       r'PCC$^{\text{def}}$',
    'pcc_s_def':     r'PCC$_{\text{S}}^{\text{def}}$',
    'pcc_d_def':     r'PCC$_{\text{D}}^{\text{def}}$',
    'pcc_p_def':     r'PCC$_{\text{P}}^{\text{def}}$',
    'pcc_od_nni':    r'PCC$^{\text{od-nni}}$',
    'pcc_s_od_nni':  r'PCC$_{\text{S}}^{\text{od-nni}}$',
    'pcc_d_od_nni':  r'PCC$_{\text{D}}^{\text{od-nni}}$',
    'pcc_p_od_nni':  r'PCC$_{\text{P}}^{\text{od-nni}}$',
}

# Variants to explicitly EXCLUDE from output (user request)
EXCLUDED_VARIANTS = {'pcc_def', 'pcc_s_def', 'pcc_d_def', 'pcc_p_def', 
                     'pcc_od_nni', 'pcc_s_od_nni', 'pcc_d_od_nni', 'pcc_p_od_nni'}

ALL_VARIANTS = list(VARIANT_LABELS.keys())

# _od variants only have pre-tuned params for these 3 datasets.
# For other datasets, _od cells will show '---' automatically since
# the CSV won't have data for them.
OD_PRETUNED_DATASETS = {'cora', 'citeseer', 'pubmed'}

# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_noise_label(noise_label):
    """
    Parse 'uniform_0.3', 'clean_0.0', 'pair_0.1', etc.
    Returns (noise_type: str, noise_rate: float).
    """
    noise_label = noise_label.strip().lower()
    # clean_0.0 or just clean
    if noise_label.startswith('clean'):
        return 'clean', 0.0
    # e.g. uniform_0.3
    m = re.match(r'^([a-z]+)_([0-9.]+)$', noise_label)
    if m:
        return m.group(1), float(m.group(2))
    raise ValueError(f'Cannot parse noise label: {noise_label!r}')


def load_our_csv(paths):
    """
    Load one or more lnpcc_results_*.csv files.
    Returns dict: {(variant, dataset, noise_type, noise_rate): (mean%, std%)}
    Mean/std in the CSV are 0-1 fractions; we convert to %.
    """
    data = {}
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset    = row['dataset'].strip().lower()
                noise_label = row['noise_rate'].strip()
                try:
                    noise_type, noise_rate = parse_noise_label(noise_label)
                except ValueError:
                    continue

                for col, val in row.items():
                    if col.endswith('_mean') and val.strip():
                        base = col[:-5]  # remove '_mean'
                        if base.startswith('lnpcc_'):
                            variant = base[6:]
                        else:
                            variant = base
                        
                        std_col = f'{base}_std'
                        try:
                            mean_val = float(val) * 100.0
                            std_val  = float(row.get(std_col, 0)) * 100.0
                        except (ValueError, TypeError):
                            continue
                        key = (variant, dataset, noise_type, noise_rate)
                        data[key] = (mean_val, std_val)
    return data


def load_noisygl_csv(path):
    """
    Load NoisyGL paper results CSV (manually constructed from paper tables).
    Format: method, dataset, noise_type, noise_rate, mean, std
    mean/std already in % (0-100).
    Returns dict: {(method, dataset, noise_type, noise_rate): (mean%, std%)}
    """
    data = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row['method'].strip().lower(),
                row['dataset'].strip().lower(),
                row['noise_type'].strip().lower(),
                float(row['noise_rate']),
            )
            try:
                data[key] = (float(row['mean']), float(row['std']))
            except (ValueError, KeyError):
                continue
    return data


# ── LaTeX generation ──────────────────────────────────────────────────────────

def fmt_cell(mean, std, bold=False):
    s = f'{mean:.2f} \\scriptscriptstyle\\pm {std:.2f}'
    s = f'$\\mathbf{{{s}}}$' if bold else f'${s}$'
    return s


def fmt_cell_nostd(mean, bold=False):
    s = f'{mean:.2f}'
    return f'\\textbf{{{s}}}' if bold else s


def make_combined_table(
    datasets,
    noise_types,
    noise_rates,
    methods,           # list of (method_id, display_label, is_lnpcc)
    our_data,          # {(variant, ds, nt, nr): (mean, std)}
    noisygl_data,      # {(method, ds, nt, nr): (mean, std)} or {}
    caption,
    label,
):
    """
    Build a single LaTeX table where rows are (Dataset, Noise Type, Noise Rate).
    Cols: Dataset | Noise Type | Noise Rate | Method1 | Method2 | ...
    """
    n_methods = len(methods)
    
    # col spec: lll + c * n_methods
    col_spec = 'lll' + 'c' * n_methods
    method_header = ' & '.join(f'\\textbf{{{m[1]}}}' for m in methods)

    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')
    lines.append(
        f'\\textbf{{Dataset}} & \\textbf{{Noise}} & \\textbf{{Rate}} & {method_header} \\\\'
    )
    lines.append(r'\midrule')

    for ds_idx, ds in enumerate(datasets):
        ds_label = DATASET_LABELS.get(ds, ds.capitalize())
        
        # Determine how many rows for this dataset
        # nt * nr
        total_ds_rows = 0
        nt_groups = []
        for nt in noise_types:
            rates = [0.0] if nt == 'clean' else [nr for nr in noise_rates if nr > 0.0]
            total_ds_rows += len(rates)
            nt_groups.append((nt, rates))

        for g_idx, (nt, rates) in enumerate(nt_groups):
            nt_label = NOISE_TYPE_LABELS.get(nt, nt.capitalize())
            
            for r_idx, nr in enumerate(rates):
                # Multirow logic
                ds_cell = f'\\multirow{{{total_ds_rows}}}{{*}}{{{ds_label}}}' if (g_idx == 0 and r_idx == 0) else ''
                nt_cell = f'\\multirow{{{len(rates)}}}{{*}}{{{nt_label}}}' if r_idx == 0 else ''
                nr_cell = 'Clean' if nr == 0.0 else f'{nr:.1f}'

                cells = []
                # Find best in this row (across methods)
                row_vals = []
                for m_id, m_label, is_lnpcc in methods:
                    key = (m_id, ds, nt, nr)
                    v = our_data.get(key) if is_lnpcc else noisygl_data.get(key)
                    row_vals.append(v)
                
                valid_means = [v[0] for v in row_vals if v is not None]
                best_mean = max(valid_means) if valid_means else None

                for v in row_vals:
                    if v is None:
                        cells.append('---')
                    else:
                        mean, std = v
                        bold = (best_mean is not None and abs(mean - best_mean) < 1e-4)
                        cells.append(fmt_cell(mean, std, bold))

                row = f'{ds_cell} & {nt_cell} & {nr_cell} & ' + ' & '.join(cells) + r' \\'
                lines.append(row)
            
            if g_idx < len(nt_groups) - 1:
                lines.append(r'\cmidrule(lr){2-' + str(n_methods + 3) + '}')

        if ds_idx < len(datasets) - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table*}')
    lines.append('')
    return '\n'.join(lines)


def make_excel_numeric(
    output_path,
    datasets,
    noise_types,
    noise_rates,
    methods,
    our_data,
    noisygl_data,
):
    """
    Generate an Excel file with numeric values and separate Mean/Std columns.
    Rows: (Dataset, Noise Type, Noise Rate)
    """
    if pd is None:
        print("ERROR: pandas not found. Excel export disabled.", file=sys.stderr)
        return

    rows = []
    for ds in datasets:
        ds_label = DATASET_LABELS.get(ds, ds.capitalize())
        for nt in noise_types:
            nt_label = NOISE_TYPE_LABELS.get(nt, nt.capitalize())
            rates = [0.0] if nt == 'clean' else [nr for nr in noise_rates if nr > 0.0]
            
            for nr in rates:
                nr_val = 0.0 if nr == 0.0 else nr
                row_data = {
                    'Dataset': ds_label,
                    'Noise Type': nt_label,
                    'Noise Rate': nr_val
                }
                
                for m_id, m_label, is_lnpcc in methods:
                    key = (m_id, ds, nt, nr)
                    v = our_data.get(key) if is_lnpcc else noisygl_data.get(key)
                    
                    # Clean up label for column headers (keep underscores for method ID)
                    base_label = m_id
                    if v is None:
                        row_data[f'{base_label}_mean'] = None
                        row_data[f'{base_label}_std'] = None
                    else:
                        row_data[f'{base_label}_mean'] = v[0]
                        row_data[f'{base_label}_std'] = v[1]
                
                rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)


# ── Driver ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate NoisyGL-style LaTeX tables from LN-PCC results'
    )
    p.add_argument('csv_files', nargs='+',
                   help='One or more lnpcc_results_*.csv files')
    p.add_argument('--noisygl_csv', type=str, default=None,
                   help='Optional CSV with NoisyGL paper results for comparison '
                        '(columns: method,dataset,noise_type,noise_rate,mean,std)')
    p.add_argument('--variants', nargs='+', default=None,
                   choices=ALL_VARIANTS,
                   help='Variants to include (default: all found in CSV)')
    p.add_argument('--noisygl_methods', nargs='+', default=None,
                   help='NoisyGL method names to include from --noisygl_csv '
                        '(e.g. gcn nrgnn rtgnn). Default: all found.')
    p.add_argument('--datasets', nargs='+', default=None,
                   help='Datasets to include (default: all found in CSV)')
    p.add_argument('--noise_types', nargs='+', default=None,
                   choices=list(NOISE_TYPE_LABELS.keys()),
                   help='Noise types to generate tables for (default: all found)')
    p.add_argument('--noise_rates', nargs='+', type=float, default=None,
                   help='Noise rates to include (default: all found)')
    p.add_argument('--output', type=str, default=None,
                   help='Output .tex file (default: print to stdout)')
    p.add_argument('--excel', type=str, default=None,
                   help='Output .xlsx file (requires pandas and openpyxl)')
    p.add_argument('--best_only', action='store_true',
                   help='Include only the single best-performing LN-PCC variant '
                        'per noise_type (determined by average rank across datasets)')
    return p.parse_args()


def best_variant(our_data, variants, datasets, noise_type, noise_rates):
    """Return the variant with the highest mean accuracy averaged across
    all (dataset, noise_rate) combinations for the given noise_type."""
    scores = defaultdict(list)
    for v in variants:
        for ds in datasets:
            for nr in noise_rates:
                val = our_data.get((v, ds, noise_type, nr))
                if val is not None:
                    scores[v].append(val[0])
    if not scores:
        return variants[0]
    return max(scores, key=lambda v: (sum(scores[v]) / len(scores[v])) if scores[v] else 0)


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    our_data = load_our_csv(args.csv_files)
    noisygl_data = load_noisygl_csv(args.noisygl_csv) if args.noisygl_csv else {}

    if not our_data:
        print('ERROR: No data loaded from CSV files.', file=sys.stderr)
        sys.exit(1)

    # ── Infer available dimensions from data ───────────────────────────
    all_variants_found = sorted({k[0] for k in our_data if k[0] not in EXCLUDED_VARIANTS})
    all_datasets_found = sorted({k[1] for k in our_data})
    all_nt_found       = {k[2] for k in our_data}
    all_nr_found       = sorted({k[3] for k in our_data})

    variants    = args.variants    or all_variants_found
    datasets    = args.datasets    or all_datasets_found
    noise_types = args.noise_types or sorted(all_nt_found,
                      key=lambda x: NOISE_TYPE_ORDER.index(x)
                      if x in NOISE_TYPE_ORDER else 99)
    noise_rates = args.noise_rates or all_nr_found

    # NoisyGL methods
    noisygl_methods_found = sorted({k[0] for k in noisygl_data}) if noisygl_data else []
    noisygl_methods = args.noisygl_methods or noisygl_methods_found

    # ── Preserve dataset order from ALL_VARIANTS / paper ──────────────
    dataset_order = ['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                     'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire']
    datasets = [d for d in dataset_order if d in datasets] + \
               [d for d in datasets if d not in dataset_order]

    print(f'Variants   : {variants}', file=sys.stderr)
    print(f'Datasets   : {datasets}', file=sys.stderr)
    print(f'Noise types: {noise_types}', file=sys.stderr)
    print(f'Noise rates: {noise_rates}', file=sys.stderr)
    if noisygl_methods:
        print(f'NoisyGL    : {noisygl_methods}', file=sys.stderr)

    # ── Build tables ───────────────────────────────────────────────────
    preamble = (
        '% Generated by make_noisygl_table.py\n'
        '% Required packages: booktabs, multirow, graphicx (for resizebox)\n'
        r'% \usepackage{booktabs, multirow, graphicx}' + '\n\n'
    )

    # ── Build methods list ─────────────────────────────────────────────
    methods = []
    for nm in noisygl_methods:
        methods.append((nm, nm.upper(), False))

    if args.best_only:
        # For simplicity, we choose best variant based on overall performance
        # if multiple noise_types/datasets are mixed.
        bv = best_variant(our_data, variants, datasets, noise_types[0], noise_rates)
        label_str = VARIANT_LABELS.get(bv, bv)
        methods.append((bv, label_str, True))
    else:
        for v in variants:
            methods.append((v, VARIANT_LABELS.get(v, v), True))

    # ── LaTeX Table ────────────────────────────────────────────────────
    caption = "Node classification accuracy (%) and standard deviation."
    tbl_label = "tab:results_all"
    
    full_output = (
        '% Generated by make_noisygl_table.py\n'
        '% Required packages: booktabs, multirow, graphicx\n'
        r'\usepackage{booktabs, multirow, graphicx}' + '\n\n'
    )
    
    full_output += make_combined_table(
        datasets=datasets,
        noise_types=noise_types,
        noise_rates=noise_rates,
        methods=methods,
        our_data=our_data,
        noisygl_data=noisygl_data,
        caption=caption,
        label=tbl_label
    )

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(full_output)
        print(f'Saved to {args.output}', file=sys.stderr)
    else:
        print(full_output)

    # ── Excel Export ───────────────────────────────────────────────────
    if args.excel:
        make_excel_numeric(
            output_path=args.excel,
            datasets=datasets,
            noise_types=noise_types,
            noise_rates=noise_rates,
            methods=methods,
            our_data=our_data,
            noisygl_data=noisygl_data
        )
        print(f'Excel saved to {args.excel}', file=sys.stderr)


if __name__ == '__main__':
    main()
