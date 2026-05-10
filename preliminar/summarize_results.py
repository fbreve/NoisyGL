import pandas as pd
import argparse
import sys
import os
import re

# ── Config ────────────────────────────────────────────────────────────────────

# Variants to explicitly EXCLUDE from output (user request)
EXCLUDED_VARIANTS = {
    'pcc_def', 'pcc_s_def', 'pcc_d_def', 'pcc_p_def', 
    'pcc_nni', 'pcc_s_nni', 'pcc_d_nni', 'pcc_p_nni'
}

DATASET_ORDER = ['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire']

def load_data(csv_files):
    """
    Load results from multiple CSV files into a tidy-format DataFrame.
    Columns: dataset, noise_type, noise_rate, method, mean, std
    """
    all_dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        
        # Identify mean columns
        mean_cols = [c for c in df.columns if c.endswith('_mean')]
        
        for col in mean_cols:
            base = col[:-5] # remove '_mean'
            if base.startswith('lnpcc_'):
                method = base[6:].replace('_od_nni', '_nni')
            else:
                method = base.replace('_od_nni', '_nni')
                
            if method in EXCLUDED_VARIANTS:
                continue
            
            # Exclude variants ending in '_od' (but NOT '_od_nni')
            if method.endswith('_od'):
                continue
                
            std_col = f'{base}_std'
            temp_df = df[['dataset', 'noise_rate', col, std_col]].copy()
            temp_df.columns = ['dataset', 'noise_rate', 'mean', 'std']
            temp_df['method'] = method
            
            # Convert values to percentages (0-1 -> 0-100)
            temp_df['mean'] *= 100.0
            temp_df['std'] *= 100.0
            
            all_dfs.append(temp_df)
    
    if not all_dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Parse noise_rate column (e.g. 'uniform_0.3' -> 'uniform', 0.3)
    def parse_noise(label):
        label = str(label).lower().strip()
        if 'clean' in label:
            return 'clean', 0.0
        m = re.match(r'^([a-z]+)_([0-9.]+)$', label)
        if m:
            return m.group(1), float(m.group(2))
        return 'unknown', 0.0

    full_df[['noise_type', 'rate']] = full_df['noise_rate'].apply(lambda x: pd.Series(parse_noise(x)))
    return full_df


def calculate_gains(df, baseline_name='gcn'):
    """
    Create a new DataFrame with (Method - Baseline) values, 
    dropping the baseline and adding AVG/MAX rows.
    """
    if baseline_name not in df.index:
        return None
    
    baseline_row = df.loc[baseline_name]
    gains = df.drop(index=baseline_name).copy()
    
    for col in df.columns:
        gains[col] = gains[col] - baseline_row[col]
    
    # Add summary rows
    if not gains.empty:
        gains.loc['MAX_GAIN'] = gains.max(axis=0)
        gains.loc['AVG_GAIN'] = gains.mean(axis=0) # Note: AVG_GAIN includes the MAX_GAIN row in mean? No, let's be careful.
        # Recalculate AVG_GAIN properly (excluding MAX_GAIN itself)
        methods_only = gains.drop(index=['MAX_GAIN', 'AVG_GAIN'], errors='ignore')
        gains.loc['AVG_GAIN'] = methods_only.mean(axis=0)
        
    return gains


def format_summary(df, existing_ds):
    """
    Calculate OVERALL_MEAN and sort by it (handling NaNs).
    """
    # Identify rows with ANY NaN in the dataset columns
    has_nan = df[existing_ds].isna().any(axis=1)
    
    # Calculate OVERALL_MEAN only for rows with NO NaNs
    df['OVERALL_MEAN'] = df[existing_ds].mean(axis=1)
    df.loc[has_nan, 'OVERALL_MEAN'] = None
    
    # Sorting logic: Non-NaN overall mean first (sorted desc), then NaN rows
    df['is_nan_row'] = has_nan
    df = df.sort_values(['is_nan_row', 'OVERALL_MEAN'], ascending=[True, False]).drop(columns=['is_nan_row'])
    
    return df


def print_to_terminal(df, title):
    print("\n" + "="*100)
    print(f" {title}")
    print("="*100)
    print(df.round(2).to_string())


def export_latex(df, path, title, is_gain=False):
    """Save summary as a LaTeX table with bold best results."""
    # Identify best in each column (only among methods, not summary rows)
    methods_idx = [i for i in df.index if i not in ['MAX_GAIN', 'AVG_GAIN']]
    best_in_col = {}
    for col in df.columns:
        valid = df.loc[methods_idx, col].dropna()
        if not valid.empty:
            best_in_col[col] = valid.max()

    with open(path, 'w', encoding='utf-8') as f:
        f.write("% " + title + "\n")
        f.write(r"\begin{table*}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\caption{" + title + "}\n")
        
        col_spec = "l" + "c" * len(df.columns)
        f.write(r"\begin{tabular}{" + col_spec + "}\n")
        f.write(r"\toprule" + "\n")
        
        headers = ["Method"] + [c.replace('_', ' ').capitalize() for c in df.columns]
        f.write(" & ".join([f"\\textbf{{{h}}}" for h in headers]) + r" \\" + "\n")
        f.write(r"\midrule" + "\n")
        
        # Track if we already added a midrule for summary rows
        added_summary_midrule = False
        
        for method, row in df.iterrows():
            if method in ['MAX_GAIN', 'AVG_GAIN'] and not added_summary_midrule:
                f.write(r"\midrule" + "\n")
                added_summary_midrule = True
            cells = [method.replace('_', r'\_')]
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    cells.append("---")
                else:
                    # Highlight if best among methods, OR if it's the MAX_GAIN row itself
                    is_best = (abs(val - best_in_col.get(col, -1e9)) < 1e-4) and (method not in ['MAX_GAIN', 'AVG_GAIN'])
                    if method == 'MAX_GAIN':
                        is_best = True # Always bold MAX_GAIN? Maybe.
                    
                    prefix = "+" if (is_gain and val > 0.005) else ""
                    s = f"{prefix}{val:.2f}"
                    cells.append(f"\\textbf{{{s}}}" if is_best else s)
            f.write(" & ".join(cells) + r" \\" + "\n")
            
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    print(f"LaTeX saved to {path}")


def add_styled_sheet(writer, df, sheet_name, is_gain=False):
    df.to_excel(writer, sheet_name=sheet_name)
    worksheet = writer.sheets[sheet_name]
    from openpyxl.styles import Font
    bold_font = Font(bold=True)
    
    # Identify best in each column
    for col_idx, col in enumerate(df.columns, start=2):
        valid = df[col].dropna()
        if not valid.empty:
            max_val = valid.max()
            for row_idx, val in enumerate(df[col], start=2):
                if not pd.isna(val) and abs(val - max_val) < 1e-4:
                    worksheet.cell(row=row_idx, column=col_idx).font = bold_font
                    
    # Auto-adjust column widths
    for idx, col in enumerate(df.columns, start=2):
        max_val_len = max([len(str(v)) for v in df[col]])
        max_len = max(max_val_len, len(str(col))) + 2
        worksheet.column_dimensions[chr(64 + idx)].width = min(max_len, 50)
    worksheet.column_dimensions['A'].width = 25


def main():
    parser = argparse.ArgumentParser(description='Summarize LN-PCC results')
    parser.add_argument('csv_files', nargs='+', help='Path to results CSV files')
    parser.add_argument('--excel', type=str, default=None, help='Save summary to Excel')
    parser.add_argument('--latex', type=str, default=None, help='Save summary to LaTeX')
    args = parser.parse_args()

    df = load_data(args.csv_files)
    if df.empty:
        print("No data found.")
        return

    # Filter out clean noise for the first table
    df_no_clean = df[df['noise_type'] != 'clean'].copy()

    # ── Processor for Scenarios ───────────────────────────────────────
    scenarios = [
        ("All Noise (No Clean)", df_no_clean, "all_noise"),
        ("Uniform 30%", df[df['noise_rate'].str.lower() == 'uniform_0.3'], "uni30"),
        ("Pair 30%", df[df['noise_rate'].str.lower() == 'pair_0.3'], "pair30"),
    ]

    results = [] # list of (title, summary_df, gain_df, tag)

    for title, scenario_df, tag in scenarios:
        if scenario_df.empty:
            print(f"\nSkipping {title}: No data.")
            continue
            
        raw_pivot = scenario_df.groupby(['method', 'dataset'])['mean'].mean().unstack('dataset')
        existing_ds = [d for d in DATASET_ORDER if d in raw_pivot.columns]
        raw_pivot = raw_pivot[existing_ds]
        
        summary = format_summary(raw_pivot.copy(), existing_ds)
        gains   = calculate_gains(summary, baseline_name='gcn')
        
        results.append((title, summary, gains, tag))
        
        print_to_terminal(summary, f"SUMMARY: {title}")
        if gains is not None:
            print_to_terminal(gains, f"RELATIVE GAIN (vs GCN): {title}")

    # ── Exports ───────────────────────────────────────────────────────
    if args.latex:
        base_latex = args.latex.replace('.tex', '')
        for title, summary, gains, tag in results:
            export_latex(summary, f"{base_latex}_{tag}.tex", f"{title} Accuracy (%)")
            if gains is not None:
                export_latex(gains, f"{base_latex}_{tag}_gain.tex", f"{title} Gain vs GCN", is_gain=True)

    if args.excel:
        if not args.excel.endswith('.xlsx'):
            args.excel += '.xlsx'
            
        with pd.ExcelWriter(args.excel, engine='openpyxl') as writer:
            for title, summary, gains, tag in results:
                add_styled_sheet(writer, summary, f"{tag[:10]} Abs")
                if gains is not None:
                    add_styled_sheet(writer, gains, f"{tag[:10]} Gain", is_gain=True)
                
        print(f"Excel saved to {args.excel}")

if __name__ == '__main__':
    main()
