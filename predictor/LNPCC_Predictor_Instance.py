from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

class lnpcc_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        # Lazy init to avoid moving to GPU during CPU phase
        self.lazy_init = True 
        super().__init__(conf, data, device)
        self._t_pcc = 0.0
        self._t_gcn = 0.0
        self._pcc_stats = None

    def method_init(self, conf, data):
        # Model init deferred until we have device slot
        self.model = None 
        self.optim = None

    def cpu_preprocess(self):
        """Phase 1: KNN cleaning and label correction on CPU."""
        t0 = time.time()
        print(f" [LNPCC-CPU] Starting KNN-based label cleaning (Parallel)...", flush=True)
        
        # We work on CPU copies of data
        labels = self.data.labels.cpu().numpy()
        noisy_labels = self.data.noisy_label.cpu().numpy()
        features = self.data.feats.cpu().numpy()
        
        # 1. KNN-based uncertainty and cleaning
        # (Using n_jobs=1 because we already parallelize across workers)
        from sklearn.neighbors import NearestNeighbors
        k = self.conf.model.get('k', 20)
        knn = NearestNeighbors(n_neighbors=k+1, n_jobs=1)
        knn.fit(features)
        distances, indices = knn.kneighbors(features)
        
        # Simple PCC logic
        new_labels = deepcopy(noisy_labels)
        kept, removed, changed = 0, 0, 0
        
        for i in range(len(labels)):
            neighbor_labels = noisy_labels[indices[i, 1:]]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred_label = unique[np.argmax(counts)]
            confidence = np.max(counts) / k
            
            if confidence > self.conf.model.get('p_grd', 0.5):
                if new_labels[i] != pred_label:
                    new_labels[i] = pred_label
                    changed += 1
                else:
                    kept += 1
            else:
                # Uncertain label
                removed += 1
        
        self.data.noisy_label = torch.from_numpy(new_labels).to(torch.long)
        self._pcc_stats = {'kept': kept, 'removed': removed, 'changed': changed}
        self._t_pcc = time.time() - t0
        print(f" [LNPCC-CPU] Done in {self._t_pcc:.1f}s. Stats: {self._pcc_stats}", flush=True)

    def train(self):
        """Phase 2: Training on GPU."""
        # Ensure model is initialized on the correct device
        if self.model is None:
            self.model = GCN(in_channels=self.conf.model['n_feat'], 
                             hidden_channels=self.conf.model['n_hidden'], 
                             out_channels=self.conf.model['n_classes'],
                             n_layers=self.conf.model['n_layer'], 
                             dropout=self.conf.model['dropout'],
                             norm_info=self.conf.model['norm_info'],
                             act=self.conf.model['act'], 
                             input_layer=self.conf.model['input_layer'],
                             output_layer=self.conf.model['output_layer']).to(self.device)
            self.optim = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])

        t0 = time.time()
        # Use the base class training loop
        res = super().train()
        self._t_gcn = time.time() - t0
        return res
