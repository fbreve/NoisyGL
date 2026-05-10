import pandas as pd

df = pd.read_csv(r'c:\Users\fbrev\Documents\Acadêmico\Simulações\Python\NoisyGL\log\instance_trials_all.csv')

# Calculate mean wall_s for each (method, dataset)
summary = df.groupby(['dataset', 'method'])['wall_s'].mean().unstack()

print("Average Wall Time (s) per Dataset and Method:")
print(summary)

# Count number of entries for each (method, dataset) to see completeness
counts = df.groupby(['dataset', 'method']).size().unstack()
print("\nNumber of trials per Dataset and Method:")
print(counts)
