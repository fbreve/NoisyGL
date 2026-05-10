import torch
import numpy as np

def cap_degree(edge_index, edge_weight=None, max_degree=400, num_nodes=None):
    """
    Caps the degree of each node in the graph by keeping only the first 'max_degree' 
    edges for each node. This is used to prevent STATUS_STACK_BUFFER_OVERRUN (0xC0000409)
    on Windows for extremely dense graphs.
    
    Parameters
    ----------
    edge_index : torch.Tensor
        The edge index tensor (2, E).
    edge_weight : torch.Tensor, optional
        The edge weight tensor (E).
    max_degree : int
        Maximum number of neighbors to keep per node.
    num_nodes : int, optional
        Total number of nodes. If None, it is inferred from edge_index.
        
    Returns
    -------
    new_edge_index : torch.Tensor
    new_edge_weight : torch.Tensor (or None)
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
        
    device = edge_index.device
    
    # Move to CPU for efficient processing if needed, but since we use numpy 
    # and this is usually called before moving to GPU in our Predictor, it's fine.
    # If it's already on GPU, we must move it.
    ei_cpu = edge_index.cpu()
    src, dst = ei_cpu[0].numpy(), ei_cpu[1].numpy()
    
    if edge_weight is not None:
        ew_cpu = edge_weight.cpu().numpy()
    else:
        ew_cpu = None
        
    # Standard degree capping logic
    new_src = []
    new_dst = []
    new_ew = [] if ew_cpu is not None else None
    
    # Find offsets for each source node if edge_index is sorted by source
    # If not sorted, we sort it.
    sort_idx = np.lexsort((dst, src))
    src = src[sort_idx]
    dst = dst[sort_idx]
    if ew_cpu is not None:
        ew_cpu = ew_cpu[sort_idx]
        
    # Get unique source nodes and their counts
    unique_src, counts = np.unique(src, return_counts=True)
    
    # We can use np.split or manual indexing to be efficient
    curr_ptr = 0
    for s_node, count in zip(unique_src, counts):
        keep = min(count, max_degree)
        new_src.append(src[curr_ptr : curr_ptr + keep])
        new_dst.append(dst[curr_ptr : curr_ptr + keep])
        if ew_cpu is not None:
            new_ew.append(ew_cpu[curr_ptr : curr_ptr + keep])
        curr_ptr += count
        
    if not new_src:
        return torch.empty((2, 0), dtype=torch.long, device=device), None
        
    final_src = np.concatenate(new_src)
    final_dst = np.concatenate(new_dst)
    final_ei = torch.from_numpy(np.stack([final_src, final_dst])).long().to(device)
    
    final_ew = None
    if new_ew:
        final_ew = torch.from_numpy(np.concatenate(new_ew)).to(device)
        
    return final_ei, final_ew
