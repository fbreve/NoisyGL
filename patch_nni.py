import os

target_file = r'C:\Users\fbrev\anaconda3\envs\noisygl\Lib\site-packages\nni\algorithms\hpo\tpe_tuner.py'

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
patched = False
for line in lines:
    if 'params = self._running_params.pop(parameter_id)' in line and not patched:
        indent = line[:line.find('params')]
        new_lines.append(f'{indent}params = self._running_params.pop(parameter_id, None)\n')
        new_lines.append(f'{indent}if params is None:\n')
        new_lines.append(f'{indent}    return\n')
        patched = True
    else:
        new_lines.append(line)

if patched:
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Patch applied successfully.")
else:
    print("Could not find the target line to patch.")
