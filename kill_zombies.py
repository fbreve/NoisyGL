import psutil
import os
import shutil

count = 0
for p in psutil.process_iter(['name', 'cmdline']):
    try:
        if p.info['name'] == 'python.exe' and p.info['cmdline']:
            cmdstr = " ".join(p.info['cmdline'])
            if 'hyperparam_opt' in cmdstr or 'single_exp' in cmdstr or 'optuna' in cmdstr:
                p.kill()
                count += 1
    except:
        pass
print(f"Killed {count} Python zombies.")

for gpu_id in [0, 1]:
    lock_dir = f"log/gpu_{gpu_id}.lock"
    if os.path.exists(lock_dir):
        shutil.rmtree(lock_dir, ignore_errors=True)
        print(f"Removed {lock_dir}")
