import psutil
import os
import time

variants = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
    try:
        cmd = " ".join(p.info['cmdline'] or []).lower()
        if "total_exp_lnpcc" in cmd:
            # find variant in cmdline
            parts = cmd.split()
            if '--variants' in parts:
                idx = parts.index('--variants')
                var = parts[idx+1]
                variants.append({'pid': p.info['pid'], 'cpu': p.info['cpu_percent'], 'variant': var})
    except:
        pass

for v in variants:
    log_file = f"./log/worker_{v['variant']}.log"
    last_banner = "N/A"
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                tmp = lines[i].strip()
                if tmp:
                    last_banner = tmp
                    break
    except:
        pass
    print(f"PID: {v['pid']:>6} | CPU: {v['cpu']:>4.1f}% | Variant: {v['variant']:<15} | Last: {last_banner[:60]}")
