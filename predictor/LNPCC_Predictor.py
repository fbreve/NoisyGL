"""
LNPCC_Predictor.py
GCN + LN-PCC predictor for the NoisyGL benchmark.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import nni
import multiprocessing as mp
from multiprocessing import shared_memory
from copy import deepcopy

from predictor.Base_Predictor import Predictor
import psutil

from filelock import FileLock

class GPULock:
    def __init__(self, device_str, timeout=7200):
        self.device_str = str(device_str)
        self.timeout = timeout
        # Match naming: gpu_cuda_0.lock
        self.lock_file = os.path.join('log', f"gpu_{self.device_str.replace(':', '_')}.lock")
        self.lock = FileLock(self.lock_file, timeout=self.timeout)
        self.acquired = False

    def __enter__(self):
        os.makedirs('log', exist_ok=True)
        try:
            self.lock.acquire()
            self.acquired = True
            return self
        except Exception as e:
            print(f" [GPULock] Failed to acquire {self.lock_file}: {e}")
            self.acquired = False
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.lock.release()
            self.acquired = False

def set_gpu_semaphore(sem):
    """Deprecated — workers now use GPULock independently."""
    pass

# ── LN-PCC imports ────────────────────────────────────────────────────────────
# Use absolute resolution so spawned workers (with different CWDs) always find
# the module regardless of where Python was launched from.
_LNPCC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'GCN+LNPCC'))
if _LNPCC_DIR not in sys.path:
    sys.path.insert(0, _LNPCC_DIR)

from lnpcc import LabelNoisePCC                              # noqa: E402
from lnpcc_graph import build_graph_from_edge_index          # noqa: E402
from lnpcc_graph import build_knn_graph, build_augmented_graph       # noqa: E402

# ── Default LN-PCC hyperparameters ────────────────────────────────────────────
_DEFAULT_P_GRD  = 0.5
_DEFAULT_K      = 10
_DEFAULT_DELTA_V    = 0.1
_DEFAULT_N_REPEATS  = 10
_DEFAULT_MAX_ITER   = 50000
_DEFAULT_ES_CHK     = 200

_KNN_VARIANTS = {'s': 'knn_same', 'd': 'knn_diff', 'p': 'knn_pure'}

def _parse_variant(variant_str, conf=None):
    """
    Parse variant string into (knn_variant, param_set).
    The 'strategy' is no longer used as we use confidence thresholds per node.
    """
    if variant_str == 'lnpcc':
        knn_key = conf.model.get('knn_mode', 'u') if conf is not None else 'u'
        return knn_key, 'od'

    s = variant_str.lower()
    if s.endswith('_nni'): s = s[:-4]
    parts = s.split('_')
    # Legacy support: parts might be [rem/rel, s/d/p, def/od] or [rem/rel, def/od]
    if len(parts) == 3:
        knn_variant = parts[1]
        param_set = parts[2]
    else:
        knn_variant = 'u'
        param_set = parts[1] if len(parts) > 1 else 'def'
    return knn_variant, param_set

class lnpcc_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        self.lazy_init = True
        super().__init__(conf, data, device)
        self.conf = conf
        self.data = data
        self.model = None
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.total_time = 0
        self.weights = None
        self.result = {}
        self._acc_test = 0.0
        self.device = torch.device(device)
        self.target_device = torch.device(device)
        self.general_init(conf, data)
        self.method_init(conf, data)

    def method_init(self, conf, data):
        variant_str = conf.model.get('method', conf.model.get('lnpcc_variant', 'lnpcc'))
        self.knn_variant, self.param_set = _parse_variant(variant_str, conf)

        self.p_grd    = float(conf.model.get('p_grd',    _DEFAULT_P_GRD))
        self.dexp     = float(conf.model.get('dexp',     2.0))
        self.k        = int(  conf.model.get('k',        _DEFAULT_K))
        self.delta_v  = float(conf.model.get('delta_v',  _DEFAULT_DELTA_V))
        self.n_repeats= int(  conf.model.get('n_repeats',_DEFAULT_N_REPEATS))
        self.es_chk   = int(  conf.model.get('es_chk',   _DEFAULT_ES_CHK))
        # Force n_jobs=1 for Phase 1 to ensure 100% stability on Windows HPO.
        # Parallelism is already provided by Optuna running multiple trials.
        self.n_jobs   = 1 # ignore conf.model.get('n_jobs')
        
        self.uniformLabeled = bool(conf.model.get('uniformLabeled', False))
        self.unc_rem = float(conf.model.get('unc_rem', 0.5))
        self.unc_rel = float(conf.model.get('unc_rel', 0.5))
        self.max_iter = int(conf.model.get('max_iter', _DEFAULT_MAX_ITER))

    def _ensure_model(self):
        if self.model is None:
            from predictor.module.GNNs import GCN
            self.model = GCN(
                in_channels    = self.conf.model['n_feat'],
                hidden_channels= self.conf.model['n_hidden'],
                out_channels   = self.conf.model['n_classes'],
                n_layers       = self.conf.model['n_layer'],
                dropout        = self.conf.model['dropout'],
                norm_info      = self.conf.model['norm_info'],
                act            = self.conf.model['act'],
                input_layer    = self.conf.model['input_layer'],
                output_layer   = self.conf.model['output_layer'],
            ).to(self.device)

    def test(self, mask):
        self._ensure_model()
        return super().test(mask)

    def evaluate(self, label, mask):
        self._ensure_model()
        return super().evaluate(label, mask)

    # Class-level cache to share KNN graphs between runs of the same dataset scenario
    _KNN_CACHE = {} # Key: (feats_id, k_nn, variant)
    _CUR_FEATS_ID = None

    def _run_lnpcc(self):
        """
        Executes LN-PCC in a separate process to prevent native crash 0xC0000409.
        """
        n_nodes = self.data.n_nodes
        # Ensure memory is C-contiguous for safe SHM transport and Cython kernels
        feats_np = np.ascontiguousarray(self.data.feats.detach().cpu().numpy())
        edge_index_np = np.ascontiguousarray(self._cpu_edge_index.numpy())
        noisy_np = self.data.labels.detach().cpu().numpy().astype(np.int64)
        if hasattr(self.data, 'noisy_label'):
            noisy_np = self.data.noisy_label.detach().cpu().numpy().astype(np.int64)
        train_mask_np = np.asarray(self.train_mask)
        
        # Prepare multiprocessing
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        
        # --- Use SharedMemory for heavy arrays to avoid Pickle Spikes on Windows ---
        shm_feats = None
        shm_adj = None
        p = None
        status = 'error'
        res = 'unknown'
        
        # ── Fully Parallel Start ──
        # Global serialization via spawn.lock removed to allow simultaneous initialization
        # on high-core-count systems.

        try:
            shm_feats = shared_memory.SharedMemory(create=True, size=feats_np.nbytes)
            shm_adj = shared_memory.SharedMemory(create=True, size=edge_index_np.nbytes)
            
            # Map and copy data
            shm_feats_arr = np.ndarray(feats_np.shape, dtype=feats_np.dtype, buffer=shm_feats.buf)
            shm_adj_arr = np.ndarray(edge_index_np.shape, dtype=edge_index_np.dtype, buffer=shm_adj.buf)
            
            shm_feats_arr[:] = feats_np[:]
            shm_adj_arr[:] = edge_index_np[:]
            
            shm_info = {
                'feats': {'name': shm_feats.name, 'shape': feats_np.shape, 'dtype': str(feats_np.dtype)},
                'adj':   {'name': shm_adj.name,   'shape': edge_index_np.shape, 'dtype': str(edge_index_np.dtype)}
            }
            
            p = ctx.Process(target=_lnpcc_worker_func, args=(
                q, n_nodes, shm_info, noisy_np, train_mask_np,
                self.k, self.knn_variant, self.p_grd, self.dexp, self.delta_v,
                self.n_repeats, self.max_iter, self.es_chk, self.uniformLabeled,
                self.unc_rel, self.unc_rem, self.n_jobs, self.conf.training.get('debug', False)
            ))
            
            p.start()
            
            # Subprocess handles the trial logic

            # Wait for result with a generous timeout
            status, res = q.get(timeout=3600)
        except Exception as e:
            import traceback
            print(f" [LN-PCC] Parent-side error or timeout: {e}")
            traceback.print_exc()
            status, res = 'error', str(e)
        finally:
            # Clean up SHM in parent
            if shm_feats: 
                shm_feats.close()
                shm_feats.unlink()
            if shm_adj: 
                shm_adj.close()
                shm_adj.unlink()
        
        if p is not None:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
            
        if status == 'success':
            return (torch.from_numpy(res['new_labels']).long(), 
                    torch.from_numpy(res['new_mask']).bool(), 
                    res['kept'], res['removed'], res['changed'])
        
        if status == 'skip':
            print(f" [LN-PCC] Skip: Only {res} classes found.")
            # Fallback to defaults for algorithm skips
            fallback_mask = np.zeros(n_nodes, dtype=bool)
            if train_mask_np.dtype == bool:
                fallback_mask[:] = train_mask_np[:]
                num_kept = int(train_mask_np.sum())
            else:
                fallback_mask[train_mask_np] = True
                num_kept = len(train_mask_np)
                
            return (torch.from_numpy(noisy_np).long(), 
                    torch.from_numpy(fallback_mask).bool(), 
                    num_kept, 0, 0)
        elif status == 'exception':
            raise RuntimeError(f"Subprocess exception:\n{res}")
        else:
            raise RuntimeError(f"Subprocess failed: {res}")

    def _run_gcn_and_test(self, train_labels, train_mask):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.optim.zero_grad()
            
            # Diagnostic for the user (visible once per trial)
            if epoch == 0 and self.conf.training.get('debug'):
                if 'cuda' in str(self.device):
                    torch.cuda.synchronize() # Final safety synchronization

            output, loss_train, acc_train = self.get_prediction(self.feats, self.edge_index, self.edge_weight, train_labels, train_mask)
            loss_train.backward()
            self.optim.step()
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            elif flag_earlystop: break
            if self.conf.training.get('debug'):
                nni.report_intermediate_result(acc_val)
                print(f"Epoch {epoch+1:05d} | Time {time.time()-t0:.4f}s | L(tr) {loss_train.item():.4f} | A(tr) {acc_train:.4f} | L(val) {loss_val:.4f} | A(val) {acc_val:.4f} {improve}", flush=True)
        
        loss_test, acc_test = self.test(self.test_mask)
        self.result['test'] = acc_test
        self._acc_test = acc_test

    def train(self):
        t_pcc = time.time()
        self._t_pcc = 0.0
        self._t_gcn = 0.0
        self._acc_test = 0.0
        self.result = {'train': -1, 'valid': -1, 'test': -1}



        # ── Step 1: Phase 1 (LN-PCC Graph Cleanup) on CPU ─────────────────────
        train_labels, train_mask, kept, removed, changed = self._run_lnpcc()
        self._t_pcc = time.time() - t_pcc

        if len(train_mask) == 0:
            train_labels = self.noisy_label
            train_mask   = self.train_mask

        # ── Phase 2: GCN (GPU, serialized via File Lock) ────────────
        gpu_id = 0
        dev_str = str(self.device)
        if 'cuda:' in dev_str: gpu_id = int(dev_str.split(':')[-1])
            
        print(f" [{self.device}] LNPCC waiting for GPU slot...", flush=True)
        t_wait_start = time.time()
        with GPULock(self.device) as lock:
            wait_duration = time.time() - t_wait_start
            self._t_wait = wait_duration
            if not lock.acquired:
                raise RuntimeError(f'[GPULock] Fatal: timeout on {self.device} after 2 hours. Aborting trial to prevent crash.')
            
            print(f" [{self.device}] Slot acquired (waited {wait_duration:.1f}s). Starting GCN...", flush=True)

            
            t_gcn_start = time.time()
            try:
                # Ensure main tensors are on target device
                self.to(self.device)

                # ── Step 2: GCN Setup (Already stabilized in Step 0) ──
                # Ensure labels and mask are on target unit
                if isinstance(train_labels, torch.Tensor): 
                    train_labels = train_labels.to(self.device)
                if isinstance(train_mask, torch.Tensor): 
                    train_mask = train_mask.to(self.device)

                # Push stabilized graph to GPU
                self.edge_index = self._cpu_edge_index.to(self.device)
                if self._cpu_edge_weight is not None:
                    self.edge_weight = self._cpu_edge_weight.to(self.device)
                else:
                    self.edge_weight = None

                self._ensure_model()

                if getattr(self, 'optim', None) is None:
                    self.optim = torch.optim.Adam(
                        self.model.parameters(),
                        lr           = self.conf.training['lr'],
                        weight_decay = self.conf.training['weight_decay'],
                    )
                
                self._run_gcn_and_test(train_labels, train_mask)
            finally:
                # Correct GPU cleanup order:
                # 1. Move all tensors/model to CPU FIRST (releases GPU tensor memory)
                # 2. Delete model reference so GC can reclaim weights
                # 3. synchronize + empty_cache AFTER move, so allocator sees freed blocks
                try:
                    self.to('cpu')
                    if hasattr(self, 'feats'): self.feats = self.feats.cpu()
                    if hasattr(self, 'edge_index'): self.edge_index = self.edge_index.cpu()
                    if hasattr(self, 'model') and self.model is not None:
                        self.model = self.model.cpu()
                        del self.model
                        self.model = None
                except:
                    pass

                if 'cuda' in str(self.device):
                    try:
                        import torch as _torch
                        import gc
                        gc.collect()
                        _torch.cuda.synchronize()
                        _torch.cuda.empty_cache()
                    except:
                        pass

                self._t_gcn = time.time() - t_gcn_start
        t_gcn = self._t_gcn
        
        # Log timing
        try:
            timing_log = os.path.join('.', 'log', 'timing_log.csv')
            os.makedirs(os.path.dirname(timing_log), exist_ok=True)
            import csv
            write_header = not os.path.exists(timing_log)
            with open(timing_log, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if write_header: w.writerow(['timestamp', 'dataset', 'device', 't_pcc_s', 't_gcn_s', 'acc_test'])
                w.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'), 
                    self.data.name, 
                    str(getattr(self, 'target_device', self.device)), 
                    f'{getattr(self, "_t_pcc", 0.0):.2f}', 
                    f'{getattr(self, "_t_gcn", 0.0):.2f}', 
                    f'{getattr(self, "_acc_test", 0.0):.4f}'
                ])
        except Exception as e:
            if self.conf.training.get('debug'):
                print(f"[LNPCC] Warning: Could not write to timing log: {e}")
        
        return self.result

def _lnpcc_worker_func(q, n_nodes, shm_info, noisy_np, train_mask, 
                      k, knn_variant, p_grd, dexp, delta_v, n_repeats, max_iter, 
                      es_chk, uniform_labeled, unc_rel, unc_rem, n_jobs, debug):
    """
    Standalone worker function to run Phase 1 (LN-PCC) in an isolated process.
    Uses ONLY NumPy to avoid GPU/CUDA library conflicts on Windows.
    """
    shm_feats = None
    shm_adj = None
    try:
        import numpy as np
        import time
        import os
        import faulthandler
        faulthandler.enable()
        
        # 0. Set up dedicated crash log
        crash_log_path = os.path.join('.', 'log', f'crash_worker_{os.getpid()}.log')
        os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
        crash_f = open(crash_log_path, 'w', encoding='utf-8')
        faulthandler.enable(file=crash_f)
        
        # Absolute isolation from native thread pools and GPU drivers
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['MKL_THREADING_LAYER'] = 'sequential'
        os.environ['KMP_AFFINITY'] = 'disabled'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['NUMBA_NUM_THREADS'] = '1'
        # Unique Numba cache per trial to avoid race conditions on Windows
        log_dir = os.path.join('.', 'log', 'numba_cache')
        os.makedirs(log_dir, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = os.path.abspath(log_dir)
        
        import threading
        # ── 1. Increase stack size for the worker THREAD (64MB) ──
        threading.stack_size(64 * 1024 * 1024)
        
        def run_pcc_thread():
            nonlocal shm_feats, shm_adj
            try:
                # ── 1. Map SharedMemory ──
                from multiprocessing import shared_memory
                shm_feats = shared_memory.SharedMemory(name=shm_info['feats']['name'])
                shm_adj = shared_memory.SharedMemory(name=shm_info['adj']['name'])
                
                feats_np = np.ndarray(shm_info['feats']['shape'], dtype=shm_info['feats']['dtype'], buffer=shm_feats.buf)
                edge_index_np = np.ndarray(shm_info['adj']['shape'], dtype=shm_info['adj']['dtype'], buffer=shm_adj.buf)

                from lnpcc import LabelNoisePCC
                from lnpcc_graph import build_graph_from_edge_index, build_knn_graph, build_augmented_graph

                train_bool = np.zeros(n_nodes, dtype=bool)
                train_bool[train_mask] = True

                # Standardization for KNN
                std = feats_np.std(axis=0)
                std[std == 0] = 1.0
                feats_std = (feats_np - feats_np.mean(axis=0)) / std

                if knn_variant != 'u':
                    neib_list_knn, neib_qt_knn = build_knn_graph(feats_std, k_nn=k, n_jobs=n_jobs)
                    neib_list_edge, neib_qt_edge = build_graph_from_edge_index(n_nodes, edge_index_np)
                    neib_list, neib_qt = build_augmented_graph(
                        neib_list_edge, neib_qt_edge, neib_list_knn, neib_qt_knn,
                        noisy_np, train_bool, strategy=knn_variant,
                    )
                else:
                    neib_list, neib_qt = build_graph_from_edge_index(n_nodes, edge_index_np)


                slabel = np.full(n_nodes, -1, dtype=np.int64)
                slabel[train_bool] = noisy_np[train_bool]

                lnpcc = LabelNoisePCC(n_jobs=n_jobs)
                lnpcc.set_graph(neib_list, neib_qt)
                
                unique_labeled = np.unique(slabel[slabel != -1])
                if len(unique_labeled) < 2:
                    if debug: print(f" [Worker-{os.getpid()}] Too few classes found: {len(unique_labeled)}", flush=True)
                    q.put(('skip', len(unique_labeled)))
                    return

                _ = lnpcc.fit_predict(
                    slabel, p_grd=p_grd, dexp=dexp, delta_v=delta_v,
                    n_repeats=n_repeats, max_iter=max_iter, es_chk=es_chk,
                    uniform_labeled=uniform_labeled,
                )
                
                owndeg = lnpcc.owndeg # (N, C)
                if owndeg is None:
                    q.put(('error', 'owndeg is None'))
                    return

                # Post-Processing
                if train_mask.dtype == bool:
                    real_train_indices = np.where(train_mask)[0]
                else:
                    real_train_indices = train_mask
                    
                owndeg_train = owndeg[real_train_indices] # (N_train, C)
                n_train = len(real_train_indices)
                
                noisy_lbl_train = noisy_np[real_train_indices]
                max_label = int(lnpcc.unique_labels.max()) if lnpcc.unique_labels.size > 0 else 0
                l2i = np.full(max_label + 1, -1, dtype=np.int64)
                for lbl, idx_c in lnpcc.label_to_idx.items():
                    l2i[lbl] = idx_c
                
                orig_lbl_idx = l2i[noisy_lbl_train]
                
                top2_order = np.argsort(owndeg_train, axis=1)[:, -2:]
                preds = top2_order[:, 1]
                
                max1 = owndeg_train[np.arange(n_train), top2_order[:, 1]]
                max2 = owndeg_train[np.arange(n_train), top2_order[:, 0]]
                
                uncertainty = np.zeros(n_train, dtype=np.float32)
                m1_nonzero = max1 > 0
                uncertainty[m1_nonzero] = max2[m1_nonzero] / max1[m1_nonzero]
                
                mask_valid_orig = (orig_lbl_idx != -1)
                is_same = (preds == orig_lbl_idx)
                
                to_relabel = (~is_same) & mask_valid_orig & (uncertainty <= unc_rel)
                to_remove = (is_same & mask_valid_orig & (uncertainty > unc_rem)) | \
                            (~is_same & mask_valid_orig & (uncertainty > unc_rel))
                
                idx_to_label_arr = np.array([lnpcc.idx_to_label[i] for i in range(lnpcc.c)], dtype=np.int64)
                
                new_labels_np = noisy_np.copy()
                if to_relabel.any():
                    new_labels_np[real_train_indices[to_relabel]] = idx_to_label_arr[preds[to_relabel]]
                    
                new_mask_bool = np.zeros(n_nodes, dtype=bool)
                new_mask_bool[real_train_indices[~to_remove]] = True
                
                q.put(('success', {
                    'new_labels': new_labels_np,
                    'new_mask': new_mask_bool,
                    'kept': int((~to_remove).sum()),
                    'removed': int(to_remove.sum()),
                    'changed': int(to_relabel.sum())
                }))
            except Exception as e_inner:
                import traceback
                msg = f"{str(e_inner)}\n{traceback.format_exc()}"
                print(f" [Worker-{os.getpid()}] CRITICAL: {msg}", flush=True)
                q.put(('exception', msg))
            finally:
                try:
                    crash_f.close()
                except Exception:
                    pass
                # Only keep crash logs that actually contain a native stack trace.
                # Empty files (0 bytes) mean the process exited cleanly — delete them.
                try:
                    if os.path.exists(crash_log_path) and os.path.getsize(crash_log_path) == 0:
                        os.remove(crash_log_path)
                except Exception:
                    pass
        
        # ── 2. Run the logic in high-stack thread ──
        worker_thread = threading.Thread(target=run_pcc_thread)
        worker_thread.start()
        worker_thread.join()
        
    except Exception as e:
        import traceback
        q.put(('exception', f"{str(e)}\n{traceback.format_exc()}"))
    finally:
        # Clean up SHM in worker
        if shm_feats: shm_feats.close()
        if shm_adj: shm_adj.close()
