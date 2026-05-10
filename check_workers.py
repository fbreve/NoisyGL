import psutil

print("Active NoisyGL Python Processes:")
print("-" * 80)
count = 0
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
    try:
        cmd = " ".join(p.info['cmdline'] or [])
        if "python" in p.info['name'].lower() and ("total_exp_lnpcc" in cmd or "run_lnpcc_parallel" in cmd or "spawn" in cmd):
            print(f"PID: {p.info['pid']:<6} | CPU: {p.info['cpu_percent']:<5.1f} | Cmd: {cmd[:100]}")
            count += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

print(f"Total relevant processes: {count}")
