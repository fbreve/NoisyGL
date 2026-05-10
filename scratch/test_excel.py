import pandas as pd
import os
path = r"c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL/scratch/NoisyGL_temp.xlsx"
print(f"Checking {path}")
if os.path.exists(path):
    print("File exists. Reading...")
    df = pd.read_excel(path, engine='openpyxl')
    print("Done reading.")
    print(df.head())
else:
    print("File does not exist.")
