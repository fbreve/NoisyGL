"""
PCC_Predictor.py
LN-PCC used as a standalone classifier (no GCN).

LN-PCC propagates labels through the graph and produces a prediction for every
node. Here we use those predictions directly as the final classification output,
without any GCN training step.

Variants (set via conf.model['pcc_variant']):
  def      — original graph, default hyperparameters
  s_def    — kNN-same augmented graph, default hyperparameters
  d_def    — kNN-diff augmented graph, default hyperparameters
  p_def    — kNN-pure augmented graph, default hyperparameters
  od_nni   — original graph, NNI-optimized hyperparameters
  s_od_nni — kNN-same augmented graph, NNI-optimized hyperparameters
  d_od_nni — kNN-diff augmented graph, NNI-optimized hyperparameters
  p_od_nni — kNN-pure augmented graph, NNI-optimized hyperparameters
"""

import sys
import os
import time
import numpy as np
import torch

from predictor.Base_Predictor import Predictor

# ── LN-PCC imports ────────────────────────────────────────────────────────────
_LNPCC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'GCN+LNPCC')
if os.path.isdir(_LNPCC_DIR):
    sys.path.insert(0, os.path.abspath(_LNPCC_DIR))

from lnpcc import LabelNoisePCC
from lnpcc_graph import build_graph_from_edge_index, build_knn_graph
try:
    from lnpcc_optuna_search import build_augmented_graph
except ImportError:
    build_augmented_graph = None

# ── Default hyperparameters ───────────────────────────────────────────────────
_DEFAULT_P_GRD   = 0.5
_DEFAULT_DEXP    = 2.0
_DEFAULT_K       = 10
_DEFAULT_DELTA_V = 0.1
_DEFAULT_N_REPEATS = 10
_DEFAULT_MAX_ITER  = 50000
_DEFAULT_ES_CHK    = 200

_KNN_VARIANTS = {'s': 'knn_same', 'd': 'knn_diff', 'p': 'knn_pure'}


def _parse_pcc_variant(variant_str):
    """
    Parse pcc_variant string into (knn_variant, param_set).
    Examples: 'def' -> (None, 'def')
              's_def' -> ('knn_same', 'def')
              'd_od_nni' -> ('knn_diff', 'od_nni')
    Strips _nni suffix for runtime logic (NNI handles param injection).
    """
    s = variant_str.lower()
    parts = s.split('_')
    # Check if first part is a kNN key
    if parts[0] in _KNN_VARIANTS:
        knn_variant = _KNN_VARIANTS[parts[0]]
        param_set   = '_'.join(parts[1:])  # e.g. 'def', 'od_nni'
    else:
        knn_variant = None
        param_set   = '_'.join(parts)      # e.g. 'def', 'od_nni'
    return knn_variant, param_set


class pcc_Predictor(Predictor):
    """
    LN-PCC standalone predictor — no GCN involved.

    train() runs LN-PCC on the graph and evaluates accuracy of its predictions
    directly against the clean labels on the test set.
    """

    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        # No model or optimizer — PCC is non-parametric
        self.model = None
        self.optim = None

        variant_str = conf.model.get('pcc_variant', 'def')
        self.knn_variant, self.param_set = _parse_pcc_variant(variant_str)

        self.p_grd     = float(conf.model.get('p_grd',     _DEFAULT_P_GRD))
        self.dexp      = float(conf.model.get('dexp',      _DEFAULT_DEXP))
        self.k         = int(  conf.model.get('k',         _DEFAULT_K))
        self.delta_v   = float(conf.model.get('delta_v',   _DEFAULT_DELTA_V))
        self.n_repeats = int(  conf.model.get('n_repeats', _DEFAULT_N_REPEATS))
        self.max_iter  = int(  conf.model.get('max_iter',  _DEFAULT_MAX_ITER))
        self.es_chk    = int(  conf.model.get('es_chk',    _DEFAULT_ES_CHK))

    def _run_pcc(self):
        """
        Runs LN-PCC and returns predictions for all nodes (numpy int64 array).
        """
        n_nodes       = self.feats.shape[0]
        feats_np      = self.feats.detach().cpu().numpy()
        edge_index_np = self.edge_index.detach().cpu().numpy()
        noisy_np      = self.noisy_label.detach().cpu().numpy().astype(np.int64)
        train_idx     = np.asarray(self.train_mask)

        train_bool = np.zeros(n_nodes, dtype=bool)
        train_bool[train_idx] = True

        std = feats_np.std(axis=0)
        std[std == 0] = 1.0
        feats_std = (feats_np - feats_np.mean(axis=0)) / std

        # ── Build graph ───────────────────────────────────────────────
        if self.knn_variant is not None:
            if build_augmented_graph is None:
                raise ImportError(
                    "lnpcc_optuna_search.build_augmented_graph is required for kNN variants."
                )
            neib_list_edge, neib_qt_edge = build_graph_from_edge_index(n_nodes, edge_index_np)
            neib_list_knn,  neib_qt_knn  = build_knn_graph(feats_std, k_nn=self.k)
            neib_list, neib_qt = build_augmented_graph(
                neib_list_edge, neib_qt_edge,
                neib_list_knn,  neib_qt_knn,
                noisy_np, train_bool,
                variant=self.knn_variant,
            )
        else:
            neib_list, neib_qt = build_graph_from_edge_index(n_nodes, edge_index_np)

        # ── Sparse labels (-1 = unlabeled) ────────────────────────────
        slabel = np.full(n_nodes, -1, dtype=np.int64)
        slabel[train_bool] = noisy_np[train_bool]

        # ── Run LN-PCC ────────────────────────────────────────────────
        lnpcc = LabelNoisePCC()
        lnpcc.set_graph(neib_list, neib_qt)
        pred = lnpcc.fit_predict(
            slabel,
            p_grd           = self.p_grd,
            dexp            = self.dexp,
            delta_v         = self.delta_v,
            n_repeats       = self.n_repeats,
            max_iter        = self.max_iter,
            es_chk          = self.es_chk,
            uniform_labeled = False,
        ).astype(np.int64)

        return pred

    def train(self):
        """
        Run LN-PCC and evaluate its predictions directly (no GCN).
        Accuracy on val/test is computed by comparing PCC predictions to clean labels.
        """
        t0 = time.time()
        pred = self._run_pcc()
        t_pcc = time.time() - t0

        clean_np = self.clean_label.detach().cpu().numpy().astype(np.int64)

        # Train accuracy (on noisy train set — how many noisy labels PCC agrees with)
        train_idx  = np.asarray(self.train_mask)
        acc_train  = float((pred[train_idx] == clean_np[train_idx]).mean())

        # Val accuracy
        val_idx   = np.asarray(self.val_mask)
        acc_val   = float((pred[val_idx] == clean_np[val_idx]).mean())

        # Test accuracy
        test_idx  = np.asarray(self.test_mask)
        acc_test  = float((pred[test_idx] == clean_np[test_idx]).mean())

        self.result['train'] = acc_train
        self.result['valid'] = acc_val
        self.result['test']  = acc_test
        self.total_time      = t_pcc

        # ── Timing log ────────────────────────────────────────────────
        try:
            timing_log = os.path.join('.', 'log', 'timing_log.csv')
            os.makedirs(os.path.dirname(timing_log), exist_ok=True)
            write_header = not os.path.exists(timing_log)
            import csv as _csv
            dataset = self.data.name
            with open(timing_log, 'a', newline='', encoding='utf-8') as _f:
                w = _csv.writer(_f)
                if write_header:
                    w.writerow(['timestamp', 'dataset', 'device', 't_pcc_s', 't_gcn_s', 'acc_test'])
                w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), dataset, str(self.target_device),
                            f'{t_pcc:.2f}', '0.00', f'{acc_test:.4f}'])
        except Exception:
            pass

        if self.conf.training.get('debug'):
            print(f"[PCC] variant={self.conf.model.get('pcc_variant')}  "
                  f"p_grd={self.p_grd:.3f}  dexp={self.dexp:.1f}  k={self.k}  "
                  f"t={t_pcc:.1f}s  val={acc_val:.4f}  test={acc_test:.4f}")

        return self.result

    # ── Override evaluate/test to work without a model ────────────────────────
    # These are called by run_single_exp for the extended_result stats.
    # We re-run PCC only once (cached in train), so we compute directly.

    def evaluate(self, label, mask):
        """Dummy evaluate — returns 0 loss and accuracy from cached PCC result."""
        return torch.tensor(0.0), self.result.get('valid', 0.0)

    def test(self, mask):
        """Returns cached test accuracy from PCC predictions."""
        return torch.tensor(0.0), self.result.get('test', 0.0)
