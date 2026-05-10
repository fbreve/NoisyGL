import torch
from utils.graph_utils import cap_degree

# Create a dummy graph with some high degree nodes
# Node 0 has 10 edges, Node 1 has 5 edges
src = torch.tensor([0]*10 + [1]*5 + [2]*2)
dst = torch.tensor(list(range(1, 11)) + list(range(12, 17)) + [18, 19])
edge_index = torch.stack([src, dst])
edge_weight = torch.ones(edge_index.shape[1])

print(f"Original edges: {edge_index.shape[1]}")
print(f"Original src: {edge_index[0]}")

# Cap degree to 3
new_ei, new_ew = cap_degree(edge_index, edge_weight, max_degree=3)

print(f"Capped edges: {new_ei.shape[1]}")
print(f"Capped src: {new_ei[0]}")
print(f"Capped weights: {new_ew}")

# Check degrees
unique, counts = torch.unique(new_ei[0], return_counts=True)
for u, c in zip(unique, counts):
    print(f"Node {u} degree: {c}")
    assert c <= 3
