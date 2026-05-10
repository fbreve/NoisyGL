import subprocess
import sys
import os
import json

# Reproduction script for amazon-ratings_clean_0.0 crash
# Scenario: clean, 0.0 noise, k=20, knn_mode='s'

params = {
    "knn_mode": "s", 
    "dexp": 3.859598596268161, 
    "p_grd": 0.21542373299543266, 
    "unc_rem": 0.8810091813591036, 
    "unc_rel": 0.6124020653386298, 
    "dropout": 0.5775405742143982, 
    "n_hidden": 32, 
    "n_layer": 3, 
    "lr": 0.005, 
    "weight_decay": 0.001, 
    "k": 20
}

params_json = json.dumps(params)

cmd = [
    sys.executable, "single_exp.py",
    "--dataset", "amazon-ratings",
    "--method", "lnpcc",
    "--noise_type", "clean",
    "--noise_rate", "0.0",
    "--device", "cuda:0", # or cpu if cuda is unavailable, but the crash was on cuda:0
    "--seed", "3000",
    "--params_json", params_json,
    '--grep_search', json.dumps({
      "name": "grep_search",
      "arguments": {
        "query": "native crash retry"
      }
    })
]

# Set environment variables for deep debugging
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["PYTHONFAULTHANDLER"] = "1"
env["OMP_NUM_THREADS"] = "1"

print(f"Running reproduction command: {' '.join(cmd)}")

proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')

try:
    for line in iter(proc.stdout.readline, ''):
        print(f"[REPRO] {line.strip()}", flush=True)
except KeyboardInterrupt:
    proc.terminate()

proc.wait()
print(f"\n[REPRO] Process finished with exit code: {proc.returncode}")
if proc.returncode == 3221226505:
    print("[REPRO] CRASH DETECTED: STATUS_STACK_BUFFER_OVERRUN (0xC0000409)")
elif proc.returncode == 3221225477:
    print("[REPRO] CRASH DETECTED: STATUS_ACCESS_VIOLATION (0xC0000005)")

