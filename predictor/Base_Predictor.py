import time
import torch
from utils.functional import accuracy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils.logger import SingleExpRecorder
from copy import deepcopy
import nni


class Predictor:
    '''
    Base class for all predictors. It provides the common training, evaluation and testing procedures.

    Parameters
    ----------
    conf : Namespace
        Method config
    data: utils.dataloader.Dataset
        Dataset used for training
    device: str
        Device to run the model on, default is 'cuda:0'.

    Returns
    -------
    None
    '''
    def __init__(self, conf, data, device='cuda:0'):
        super(Predictor, self).__init__()
        self.conf = conf
        self.data = data
        self.model = None
        self.device = torch.device(device)
        self.target_device = torch.device(device) # Preserve target device for logging
        self._t_cpu = 0.0
        self._t_gpu = 0.0
        self._t_wait = 0.0
        
        self.general_init(conf, data)
        self.method_init(conf, data)
        
        # Stability: defer moving to GPU for parallel environments
        # If lazy_init is True, the child class or launcher must call .to(device) explicitly inside a lock.
        if not getattr(self, 'lazy_init', False):
            self.to(self.device)

    def general_init(self, conf, data):
        '''
        This conducts necessary operations for an experiment, including the setting specified split,
        variables to record statistics.

        Parameters
        ----------
        conf : Namespace
            Method config
        data: utils.dataloader.Dataset
            Dataset used for training

        Returns
        -------
        None
        '''

        self.loss_fn = F.binary_cross_entropy_with_logits if data.n_classes == 1 else F.cross_entropy
        self.metric = roc_auc_score if data.n_classes == 1 else accuracy
        
        self.adj = data.adj
        
        # Stability fix: Extract indices and weights ONCE during init on CPU.
        # This avoids repeated unstable sparse operations in parallel trials.
        try:
            self._cpu_edge_index = self.adj.indices().clone().detach()
        except:
            # Fallback if adj is not a standard sparse tensor
            self._cpu_edge_index = getattr(self.data, 'edge_index', None)

        try:
            self._cpu_edge_weight = self.adj.values().clone().detach()
        except:
            self._cpu_edge_weight = getattr(self.data, 'edge_weight', None)
        
        # The GPU versions will be populated by self.to()
        self.edge_index = None
        self.edge_weight = None
        
        self.recoder = SingleExpRecorder(self.conf.training['patience'], self.conf.training['criterion'])
        self.feats = data.feats
        self.n_nodes = data.n_nodes
        self.n_classes = data.n_classes
        self.clean_label = data.labels
        self.noisy_label = getattr(data, 'noisy_label', data.labels)
        self.train_mask = data.train_masks
        self.val_mask = data.val_masks
        self.test_mask = data.test_masks
        self.result = {'train': -1, 'valid': -1, 'test': -1}
        self.weights = None
        self.start_time = time.time()
        self.total_time = -1
        self._t_cpu = 0.0
        self._t_gpu = 0.0

    def to(self, device):
        """Move essential tensors to the GPU/Device while keeping sparse structures on CPU."""
        self.device = torch.device(device)
        
        # 1. Ensure CPU cache exists (usually done in general_init, but safety check)
        if self._cpu_edge_index is None:
            self._cpu_edge_index = self.adj.indices().clone().detach()
            try:
                self._cpu_edge_weight = self.adj.values().clone().detach()
            except:
                self._cpu_edge_weight = None

        self.edge_index = self._cpu_edge_index.to(self.device)
        if self._cpu_edge_weight is not None:
            self.edge_weight = self._cpu_edge_weight.to(self.device)
        else:
            self.edge_weight = None

        # 2. Move the model
        if self.model is not None:
            self.model.to(self.device)

        # 3. Move other data attributes (feats, labels)
        self.feats = self.data.feats.to(self.device)
        self.clean_label = self.data.labels.to(self.device)
        # Some predictors might not have noisy_label set yet, or it's in data
        self.noisy_label = getattr(self.data, 'noisy_label', self.noisy_label)
        if isinstance(self.noisy_label, torch.Tensor):
            self.noisy_label = self.noisy_label.to(self.device)

        # 4. Move masks if they are tensors (some datasets use arrays/lists initially)
        def to_dev(m):
            return m.to(self.device) if isinstance(m, torch.Tensor) else m
        
        self.train_mask = to_dev(self.data.train_masks)
        self.val_mask   = to_dev(self.data.val_masks)
        self.test_mask  = to_dev(self.data.test_masks)

        # 5. Keep adj on CPU to avoid Windows/CUDA Sparse COO instability
        self.adj = self.data.adj.cpu()
        # Create a CSR version for models that prefer it
        try:
            self.adj_csr = self.adj.to_sparse_csr().to(self.device)
        except:
            self.adj_csr = self.adj.to(self.device)
        
        if self.device.type == 'cpu':
            torch.cuda.empty_cache()
            
        return self
        
    def cpu_preprocess(self):
        """Hook for CPU-bound preprocessing before GPU training starts."""
        pass

    def method_init(self, conf, data):
        '''
        This sets module and other members, which is overwritten for each method.

        Parameters
        ----------
        conf : Namespace
            Method config
        data: utils.dataloader.Dataset
            Dataset used for training

        Returns
        -------
        None
        '''
        self.model = None
        self.optim = None
        return None

    def get_prediction(self, features, edge_index, edge_weight=None, label=None, mask=None):
        '''
        This is the common training procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        features: torch.tensor
            node feature
        edge_index: torch.tensor
            graph adjacency info (2, E)
        edge_weight: torch.tensor, optional

        Returns
        -------
        output : torch.tensor
            The output of the model.
        loss : torch.tensor or None
            The loss value if label and mask are provided, otherwise None.
        acc : float or None
            The value of metric if label and mask are provided, otherwise None.
        '''
        output = self.model(features, edge_index, edge_weight)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = self.loss_fn(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        '''
        This is the common training procedure, which is overwritten for special learning procedure.

        Parameters
        ----------
        None

        Returns
        -------
        result : dict
            A dict containing train, valid and test metrics.
        '''

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            
            if self.model is None:
                print(f"[ERROR] Trial failed: self.model is None in {self.__class__.__name__}. Check initialization logic.", flush=True)
                return self.result

            self.model.train()
            self.optim.zero_grad()
            features, edge_index, edge_weight = self.feats, self.edge_index, self.edge_weight
            
            # Forward & Backward (GPU time)
            t_start_gpu = time.time()
            output, loss_train, acc_train = self.get_prediction(features, edge_index, edge_weight, self.noisy_label, self.train_mask)
            loss_train.backward()
            self.optim.step()
            self._t_gpu += (time.time() - t_start_gpu)

            # Evaluate (GPU time)
            t_start_eval = time.time()
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask)
            self._t_gpu += (time.time() - t_start_eval)

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve), flush=True)

        t_start_test = time.time()
        loss_test, acc_test = self.test(self.test_mask)
        self._t_gpu += (time.time() - t_start_test)
        self.result['test'] = acc_test
        
        if self.conf.training['debug']:
            print('Optimization Finished!')
            print('Time(s): {:.4f}'.format(self.total_time))
            print("Loss(test) {:.4f} | Acc(test) {:.4f}".format(loss_test.item(), acc_test))
        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        features, edge_index, edge_weight = self.feats, self.edge_index, self.edge_weight
        with torch.no_grad():
            _, loss, acc = self.get_prediction(features, edge_index, edge_weight, label, mask)
        return loss, acc

    def test(self, mask):
        '''
        This is the common test procedure, which is overwritten for special test procedure.

        Returns
        -------
        loss : float
            Test loss.
        metric : float
            Test metric.
        '''
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        label = self.clean_label
        return self.evaluate(label, mask)







