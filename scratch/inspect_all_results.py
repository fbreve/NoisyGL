import pandas as pd
import os

file_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
if os.path.exists(file_path):
    df = pd.read_excel(file_path, sheet_name=0)
    print("--- NoisyGL.xlsx (Sheet 0) ---")
    print(df.iloc[:40, :].to_string())
else:
    print("NoisyGL.xlsx not found")

file_path_v2 = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\results.xlsx'
if os.path.exists(file_path_v2):
    df2 = pd.read_excel(file_path_v2, sheet_name=0)
    print("\n--- results.xlsx (Sheet 0) ---")
    print(df2.iloc[:40, :].to_string())
else:
    print("results.xlsx not found")
