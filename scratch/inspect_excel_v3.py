import pandas as pd
import os

file_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
df = pd.read_excel(file_path, sheet_name=0, header=1) # Using row 1 as header
print("Columns:")
print(df.columns.tolist())
print("\nFirst 10 rows:")
print(df.head(10).to_string())
