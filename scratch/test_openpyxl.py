import openpyxl
path = r"c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL/scratch/NoisyGL_temp.xlsx"
print(f"Opening workbook {path}")
wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
print("Workbook opened.")
sheet = wb.active
print(f"Sheet title: {sheet.title}")
for row in sheet.iter_rows(max_row=5):
    print([cell.value for cell in row])
