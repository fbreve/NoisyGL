import psutil
import sys
import os

KEYWORDS = ['run_lnpcc_parallel.py', 'total_exp_lnpcc.py', 'single_exp.py', 'hyperparam_opt_optuna.py']

def lower_priority():
    print("Searching for active simulation processes...")
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline:
                continue
            
            # Join cmdline to a single string for keyword searching
            cmd_str = " ".join(cmdline)
            
            if any(kw in cmd_str for kw in KEYWORDS):
                print(f"Found process: {proc.info['pid']} - {proc.info['name']}")
                print(f"  Cmd: {cmd_str[:100]}...")
                
                # Set priority to Below Normal
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                print(f"  -> Priority set to BELOW_NORMAL")
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if count == 0:
        print("No matching simulation processes found.")
    else:
        print(f"\nDone! Adjusted priority for {count} processes.")

if __name__ == '__main__':
    lower_priority()
