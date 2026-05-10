import pandas as pd
import os

file_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
if os.path.exists(file_path):
    xls = pd.ExcelFile(file_path)
    print(f"Sheets: {xls.sheet_names}")
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        print(f"\nSheet: {sheet}")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(2))
else:
    print("File not found")
