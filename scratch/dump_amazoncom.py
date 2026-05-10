import pandas as pd
import os

XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
df = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)

# Find where "Amazon-C" or "amazoncom" is
print("Searching for Amazon-C or amazoncom...")
for i in range(len(df)):
    for j in range(len(df.columns)):
        val = str(df.iloc[i, j]).lower()
        if 'amazon-c' in val or 'amazoncom' in val:
            print(f"Found at Row {i}, Col {j}: {df.iloc[i, j]}")
            # Print the whole block (next 16 rows)
            print(df.iloc[i:i+17, :].to_string())
            break
