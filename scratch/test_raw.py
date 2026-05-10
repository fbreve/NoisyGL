path = r"c:/Users/fbrev/Documents/Acadêmico/Simulações/Python/NoisyGL/scratch/NoisyGL_temp.xlsx"
with open(path, "rb") as f:
    data = f.read(100)
    print(f"Read {len(data)} bytes: {data}")
