import pd_helper # If available, but I'll use standard pandas
import pandas as pd
import os

file_path = r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\results\NoisyGL.xlsx'
if os.path.exists(file_path):
    df = pd.read_excel(file_path, sheet_name=0)
    print(df.iloc[:20, :].to_string())
else:
    print("File not found")
