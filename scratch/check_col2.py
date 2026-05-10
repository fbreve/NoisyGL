import pandas as pd
import os

XLSX_PATH = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
df = pd.read_excel(XLSX_PATH, sheet_name=0, header=None)

print("Column 2 values:")
for i, val in enumerate(df[2]):
    if pd.notna(val) and str(val).strip() != "":
        print(f"Row {i}: {val}")
