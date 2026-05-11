import json
import os

db_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\hpo_db_instance.json'

if not os.path.exists(db_path):
    print(f"File not found: {db_path}")
    exit()

with open(db_path, 'r') as f:
    db = json.load(f)

initial_count = len(db)
# Remove entries where retest_avg is 0.0 (indicating failure)
# and those that have "pignn" and "amazoncom" specifically if they failed
keys_to_remove = []
for key, val in db.items():
    if val.get('retest_avg') == 0.0:
        keys_to_remove.append(key)

for key in keys_to_remove:
    print(f"Removing failed entry: {key}")
    del db[key]

final_count = len(db)
print(f"Removed {initial_count - final_count} entries.")

with open(db_path, 'w') as f:
    json.dump(db, f, indent=2)

# Also clean the .done files so the launcher runs them again
done_dir = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\instance_done'
if os.path.exists(done_dir):
    for key in keys_to_remove:
        done_file = os.path.join(done_dir, f"{key}.done")
        if os.path.exists(done_file):
            print(f"Removing done file: {done_file}")
            os.remove(done_file)
