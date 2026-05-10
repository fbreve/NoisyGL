"""
monitor_gpus.py
Monitora uso de GPU(s) e CPU em tempo real enquanto o benchmark roda.
Imprime no terminal e salva CSV em ./log/monitor_<timestamp>.csv

Uso:
    python monitor_gpus.py                    # monitora todas as GPUs, intervalo 5s
    python monitor_gpus.py --gpus 0 1         # só GPUs 0 e 1
    python monitor_gpus.py --interval 10      # a cada 10 segundos
    python monitor_gpus.py --no_print         # só salva CSV, não imprime

Rodar em paralelo ao benchmark:
    # Terminal 1
    python monitor_gpus.py --gpus 0 1

    # Terminal 2
    python run_lnpcc_parallel.py ... --skip_nni --resume
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print('[monitor] WARNING: psutil not installed. CPU monitoring disabled.')
    print('[monitor]          Install with: pip install psutil')


def query_nvidia_smi(gpu_ids):
    """
    Query nvidia-smi for GPU stats.
    Returns list of dicts with keys: id, name, util_pct, mem_used_mb, mem_total_mb, temp_c, power_w
    """
    query_fields = 'index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
    try:
        result = subprocess.run(
            ['nvidia-smi',
             f'--query-gpu={query_fields}',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            continue
        try:
            gpu_id = int(parts[0])
            if gpu_ids and gpu_id not in gpu_ids:
                continue
            gpus.append({
                'id':           gpu_id,
                'name':         parts[1],
                'util_pct':     float(parts[2]) if parts[2] not in ('[N/A]', 'N/A') else 0.0,
                'mem_used_mb':  float(parts[3]) if parts[3] not in ('[N/A]', 'N/A') else 0.0,
                'mem_total_mb': float(parts[4]) if parts[4] not in ('[N/A]', 'N/A') else 0.0,
                'temp_c':       float(parts[5]) if parts[5] not in ('[N/A]', 'N/A') else 0.0,
                'power_w':      float(parts[6]) if parts[6] not in ('[N/A]', 'N/A', '[Not Supported]') else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return gpus


def get_cpu_stats():
    if not _HAS_PSUTIL:
        return {'cpu_pct': 0.0, 'ram_used_gb': 0.0, 'ram_total_gb': 0.0}
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    return {
        'cpu_pct':      cpu_pct,
        'ram_used_gb':  mem.used  / 1024**3,
        'ram_total_gb': mem.total / 1024**3,
    }


def print_row(ts, cpu, gpus, first=False):
    if first:
        header = f"{'Time':19}  {'CPU%':>5}  {'RAM':>9}"
        for g in gpus:
            header += f"  GPU{g['id']}({g['name'].split()[-1]}):util%  mem(MB)   temp  power"
        print(header)
        print('-' * len(header))

    row = f"{ts}  {cpu['cpu_pct']:>5.1f}  {cpu['ram_used_gb']:>6.1f}GB"
    for g in gpus:
        mem_pct = 100 * g['mem_used_mb'] / max(g['mem_total_mb'], 1)
        row += (f"  {g['util_pct']:>6.1f}%"
                f"  {g['mem_used_mb']:>5.0f}/{g['mem_total_mb']:.0f}"
                f"  {g['temp_c']:>4.0f}C"
                f"  {g['power_w']:>5.0f}W")
    print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description='GPU + CPU monitor for NoisyGL benchmark')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='GPU IDs to monitor (default: all)')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Sampling interval in seconds (default: 5)')
    parser.add_argument('--no_print', action='store_true',
                        help='Only write CSV, do not print to terminal')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: ./log/monitor_<ts>.csv)')
    args = parser.parse_args()

    gpu_ids = set(args.gpus) if args.gpus else set()

    ts_start = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    csv_path = args.output or os.path.join(log_dir, f'monitor_{ts_start}.csv')

    print(f'[monitor] Logging to {csv_path}  (interval={args.interval}s)')
    print(f'[monitor] Press Ctrl+C to stop.\n')

    # Warm up psutil CPU (first call returns 0)
    if _HAS_PSUTIL:
        psutil.cpu_percent(interval=None)

    fieldnames = None
    first_print = True
    sample = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = None

        try:
            while True:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cpu  = get_cpu_stats()
                gpus = query_nvidia_smi(gpu_ids)

                # Build CSV row
                row = {'timestamp': ts,
                       'cpu_pct': cpu['cpu_pct'],
                       'ram_used_gb': round(cpu['ram_used_gb'], 2),
                       'ram_total_gb': round(cpu['ram_total_gb'], 2)}
                for g in gpus:
                    pfx = f"gpu{g['id']}"
                    row[f'{pfx}_util_pct']    = g['util_pct']
                    row[f'{pfx}_mem_used_mb'] = g['mem_used_mb']
                    row[f'{pfx}_mem_total_mb']= g['mem_total_mb']
                    row[f'{pfx}_temp_c']      = g['temp_c']
                    row[f'{pfx}_power_w']     = g['power_w']

                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(row)
                csvfile.flush()

                if not args.no_print:
                    print_row(ts, cpu, gpus, first=first_print)
                    first_print = False

                sample += 1
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f'\n[monitor] Stopped after {sample} samples. CSV saved: {csv_path}')

    # Summary
    if sample > 0:
        print(f'\n[monitor] Summary: {sample} samples over ~{sample*args.interval/60:.1f} min')
        print(f'[monitor] To analyze: load {csv_path} in pandas or Excel.')


if __name__ == '__main__':
    main()
