import torch
import numpy as np
import time
from utils.labelnoise import add_instance_dependent_label_noise

def test_performance():
    # Simulate a large dataset (similar to Roman Empire or bigger)
    num_nodes = 30000
    feature_size = 300
    num_classes = 10
    noise_rate = 0.3
    
    # Random data
    feature = torch.randn(num_nodes, feature_size).cuda()
    labels = np.random.randint(0, num_classes, num_nodes)
    
    print(f"Testing vectorized instance noise for {num_nodes} nodes...")
    t0 = time.time()
    noisy_labels = add_instance_dependent_label_noise(
        noise_rate=noise_rate,
        feature=feature,
        labels=labels,
        num_classes=num_classes,
        norm_std=0.1,
        seed=42
    )
    t1 = time.time()
    
    print(f"Done! Noise generation took: {t1 - t0:.4f}s")
    
    actual_rate = (noisy_labels != labels).mean()
    print(f"Actual noise rate: {actual_rate:.4f} (Target: {noise_rate})")
    
    if t1 - t0 < 1.0:
        print("SUCCESS: Performance is excellent (sub-second for 30k nodes).")
    else:
        print("WARNING: Performance is slower than expected.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_performance()
    else:
        print("CUDA not available, skipping GPU performance test.")
