import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
from utils.tools import setup_seed
from scipy import stats


def uniform_noise_cp(n_classes, noise_rate):
    '''
    Generate a uniform noise corruption probability matrix.
    Parameters
    ----------
    n_classes: int
        The number of label classes
    noise_rate: float
        The noise rate, which is the probability of a label being flipped to another label.

    Returns
    -------
    P: np.ndarray
        Corruption probability matrix, where P[i, j] is the probability of label i being flipped to label j.
    '''
    P = np.float64(noise_rate) / np.float64(n_classes - 1) * np.ones((n_classes, n_classes))
    np.fill_diagonal(P, (np.float64(1) - np.float64(noise_rate)) * np.ones(n_classes))
    diag_idx = np.arange(n_classes)
    P[diag_idx, diag_idx] = P[diag_idx, diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def pair_noise_cp(n_classes, noise_rate):
    '''
    Generate a pairwise noise corruption probability matrix.
    Parameters
    ----------
    n_classes: int
        The number of label classes
    noise_rate: float
        The noise rate, which is the probability of a label being flipped to another label.

    Returns
    -------
    P: np.ndarray
        Corruption probability matrix, where P[i, j] is the probability of label i being flipped to label j.
    '''
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        P[i, i - 1] = np.float64(noise_rate)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def random_noise_cp(n_classes, noise_rate):
    '''
    Generate a random noise corruption probability matrix.
    Parameters
    ----------
    n_classes: int
        The number of label classes
    noise_rate: float
        The noise rate, which is the probability of a label being flipped to another label.
    Returns
    -------
    P: np.ndarray
        Corruption probability matrix, where P[i, j] is the probability of label i being flipped to label j.

    '''
    P = (1.0 - np.float64(noise_rate)) * np.eye(n_classes)
    for i in range(n_classes):
        tp = np.random.rand(n_classes)
        tp[i] = 0
        tp = (tp / tp.sum()) * noise_rate
        P[i, :] += tp
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def add_instance_independent_label_noise(labels, cp, random_seed):
    '''
    Add instance-independent label noise to the labels (Vectorized).
    '''
    n_labels = labels.shape[0]
    setup_seed(random_seed)
    
    # Get probabilities for each node based on its current label
    probs = torch.from_numpy(cp[labels]).float()
    
    # Sample new labels in one batch
    noisy_labels = torch.multinomial(probs, num_samples=1).view(-1).numpy()
    
    return noisy_labels


def add_instance_dependent_label_noise(noise_rate, feature, labels, num_classes, norm_std, seed):
    '''
    Add instance-dependent label noise to the labels (Memory Efficient).
    '''
    label_num = num_classes
    setup_seed(seed)
    num_nodes = labels.shape[0]
    feature_size = feature.shape[1]

    # 1. Generate flip rates for each node (CPU part, but fast enough for 90k)
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std,
                                        (1 - noise_rate) / norm_std,
                                        loc=noise_rate,
                                        scale=norm_std)
    flip_rate = torch.from_numpy(flip_distribution.rvs(num_nodes)).float().to(feature.device)
    
    # 2. Labels as tensor
    labels_tensor = torch.from_numpy(labels).to(torch.long).to(feature.device)
    
    # 3. Weights W: [C, D, C]
    W = torch.randn(label_num, feature_size, label_num, device=feature.device)

    # 4. Compute probabilities P: [N, C] efficiently
    # We group nodes by label to avoid expanding W into [N, D, C] (saves GBs of VRAM)
    P = torch.zeros(num_nodes, label_num, device=feature.device)
    
    for c in range(label_num):
        mask = (labels_tensor == c)
        if not mask.any():
            continue
            
        # Get features for nodes in this class
        feat_c = feature[mask] # [N_c, D]
        
        # Compute activations A_c = x @ W[c]
        A_c = feat_c.mm(W[c]) # [N_c, C]
        
        # Mask the true class
        A_c[:, c] = -1e10
        
        # Softmax for others
        probs_other_c = F.softmax(A_c, dim=1)
        
        # Apply flip rates
        P[mask] = flip_rate[mask].unsqueeze(1) * probs_other_c
        P[mask, c] = 1.0 - flip_rate[mask]
    
    # 5. Sample noisy labels
    new_label = torch.multinomial(P, 1).view(-1).cpu().numpy()
    
    return new_label


def label_process(labels, features, n_classes, noise_type='uniform', noise_rate=0,
                  random_seed=5, debug=True):
    '''
    Parameters
    ----------
    labels: np.ndarray (torch.Tensor on caller side)
        Original labels
    features: torch.Tensor
        Node features
    n_classes: int
        The number of label classes
    noise_type: string
        Specify the type of label noise
    noise_rate: float
        Specify label noise rate
    random_seed: int
        Set random seed
    debug: bool
        Debug mode

    Returns
    -------
    noisy_train_labels: torch.Tensor
        Processed noisy labels
    modified_mask: np.ndarray
        Indices of modified labels

    '''
    setup_seed(random_seed)
    assert (noise_rate >= 0.) and (noise_rate <= 1.)

    if debug:
        print('----label noise information:------')

    # Escolhe a matriz de corrupção para tipos independentes
    if noise_rate > 0.0:
        if noise_type == 'clean':
            if debug:
                print("Clean data")
            cp = np.eye(n_classes)
        elif noise_type == 'uniform':
            if debug:
                print("Uniform noise")
            cp = uniform_noise_cp(n_classes, noise_rate)
        elif noise_type == 'random':
            if debug:
                print("Random noise")
            cp = random_noise_cp(n_classes, noise_rate)
        elif noise_type == 'pair':
            if debug:
                print("Pair noise")
            cp = pair_noise_cp(n_classes, noise_rate)
        elif noise_type == 'instance':
            # instance-dependent não usa cp
            cp = None
        else:
            cp = np.eye(n_classes)
            if debug:
                print("Invalid noise type for a non-zero noise rate: " + noise_type)
    else:
        cp = np.eye(n_classes)

    # Aplica o ruído de fato
    if noise_rate > 0.0:
        if noise_type in ['clean', 'uniform', 'random', 'pair']:
            # Apenas ruído independente de instância
            noisy_labels = add_instance_independent_label_noise(
                labels.cpu().numpy(), cp, random_seed
            )
        elif noise_type == 'instance':
            if debug:
                print("Instance dependent noise")
            noisy_labels = add_instance_dependent_label_noise(
                noise_rate=noise_rate,
                feature=features,
                labels=labels.cpu().numpy(),
                num_classes=n_classes,
                norm_std=0.1,
                seed=random_seed
            )
        else:
            # fallback: sem alteração
            noisy_labels = labels.cpu().numpy()
    else:
        if debug:
            print('Clean data')
        noisy_labels = labels.cpu().numpy()

    noisy_train_labels = torch.tensor(noisy_labels).to(torch.long).to(labels.device)

    # Calcula taxa real de ruído via tensores (mais rápido)
    diff = (noisy_train_labels != labels)
    actual_noise_rate = diff.float().mean().item()
    modified_mask = torch.where(diff)[0].cpu().numpy()

    if debug:
        print('#Actual noise rate %.2f ' % actual_noise_rate)

    return noisy_train_labels, modified_mask
