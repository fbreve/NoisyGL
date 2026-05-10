from predictor.Base_Predictor import Predictor
from predictor.module.NRGNN import EstimateAdj
from predictor.module.GNNs import GCN
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import torch_geometric.utils as utils
import numpy as np


class nrgnn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.predictor = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                             out_channels=conf.model['n_classes'],
                             n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                             norm_info=conf.model['norm_info'],
                             act=conf.model['act'], input_layer=conf.model['input_layer'],
                             output_layer=conf.model['output_layer']).to(self.device)

        self.model = GCN(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                         out_channels=conf.model['n_classes'],
                         n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                         norm_info=conf.model['norm_info'],
                         act=conf.model['act'], input_layer=conf.model['input_layer'],
                         output_layer=conf.model['output_layer']).to(self.device)

        self.estimator = EstimateAdj(conf).to(self.device)

        self.optim = torch.optim.Adam(
            list(self.model.parameters()) + list(self.estimator.parameters()) + list(self.predictor.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

        self.best_pred = None
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_acc_pred_val = 0
        
        edge_index = self._cpu_edge_index.to(self.device)
        features = self.feats
        
        # Safe handling of train_mask
        t_mask = self.train_mask
        t_mask_np = t_mask.cpu().numpy() if isinstance(t_mask, torch.Tensor) else t_mask
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(t_mask_np))).to(self.device)

        t_cpu_start = time.time()
        self.pred_edge_index = self.get_train_edge(edge_index, features, self.conf.model['n_p'], self.train_mask)
        self._t_cpu += (time.time() - t_cpu_start)

    def get_prediction(self, features, edge_index, label=None, mask=None, reformed_adj_model=None):
        if reformed_adj_model is None:
            representations, rec_loss = self.estimator(edge_index, features)
            predictor_weights = self.estimator.get_estimated_weigths(self.pred_edge_index, representations)
            pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
            predictor_weights_1 = torch.cat([torch.ones([edge_index.shape[1]], device=self.device), predictor_weights],
                                            dim=0).detach()
            try:
                reformed_adj_pred = torch.sparse_coo_tensor(pred_edge_index, predictor_weights_1, [self.n_nodes, self.n_nodes]).to_sparse_csr()
            except:
                reformed_adj_pred = torch.sparse_coo_tensor(pred_edge_index, predictor_weights_1, [self.n_nodes, self.n_nodes])
            log_pred = self.predictor(features, reformed_adj_pred)
            if self.best_pred == None:
                pred = F.softmax(log_pred, dim=1).detach()
                self.best_pred = pred
                t_cpu_start = time.time()
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(self.best_pred)
                self._t_cpu += (time.time() - t_cpu_start)
            else:
                pred = self.best_pred
            estimated_weights = self.estimator.get_estimated_weigths(self.unlabel_edge_index, representations)
            estimated_weights_1 = torch.cat([predictor_weights_1, estimated_weights], dim=0).detach()
            model_edge_index = torch.cat([pred_edge_index, self.unlabel_edge_index], dim=1)
            try:
                reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights_1,
                                                         [self.n_nodes, self.n_nodes]).to_sparse_csr()
            except:
                reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights_1,
                                                         [self.n_nodes, self.n_nodes])
            output = self.model(features, reformed_adj_model)
            pred_model = F.softmax(output, dim=1)
            eps = 1e-8
            pred_model = pred_model.clamp(eps, 1 - eps)
            loss_add = (-torch.sum(pred[self.idx_add] * torch.log(pred_model[self.idx_add]), dim=1)).mean()
            loss_pred = self.loss_fn(log_pred[mask], label[mask])
            loss_gcn = self.loss_fn(output[mask], label[mask])
            loss = loss_gcn + loss_pred + self.conf.model['alpha'] * rec_loss + self.conf.model['beta'] * loss_add
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc, estimated_weights_1, model_edge_index, predictor_weights_1, pred
        else:
            output = self.model(features, reformed_adj_model)
            loss_gcn = self.loss_fn(output[mask], label[mask])
            loss = loss_gcn
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
            return output, loss, acc

    def train(self):
        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.predictor.train()
            self.optim.zero_grad()
            features, edge_index = self.feats, self.edge_index
            
            t_gpu_start = time.time()
            _, loss_train, acc_train, estimated_weights, model_edge_index, predictor_weights, pred \
                = self.get_prediction(features, edge_index, self.noisy_label, self.train_mask)
            loss_train.backward()
            self.optim.step()
            self._t_gpu += (time.time() - t_gpu_start)

            t_gpu_eval = time.time()
            loss_val, acc_val = self.evaluate(self.noisy_label, self.val_mask, model_edge_index, estimated_weights)
            self._t_gpu += (time.time() - t_gpu_eval)

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.best_graph = estimated_weights.detach()
                self.best_model_index = model_edge_index
                self.weights = deepcopy(self.model.state_dict())
                self.best_pred_graph = predictor_weights.detach()
                self.best_pred = pred.detach()
                t_cpu_start = time.time()
                self.unlabel_edge_index, self.idx_add = self.get_model_edge(pred)
                self._t_cpu += (time.time() - t_cpu_start)
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve), flush=True)
        
        t_gpu_test = time.time()
        loss_test, acc_test = self.test(self.test_mask)
        self._t_gpu += (time.time() - t_gpu_test)
        self.result['test'] = acc_test
        return self.result

    def evaluate(self, label, mask, model_edge_index=None, estimated_weights=None):
        features = self.feats
        if model_edge_index is None:
            estimated_weights = self.best_graph
            model_edge_index = self.best_model_index
        self.model.eval()
        self.predictor.eval()
        with torch.no_grad():
            try:
                reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights,
                                                             [self.n_nodes, self.n_nodes]).to_sparse_csr()
            except:
                reformed_adj_model = torch.sparse_coo_tensor(model_edge_index, estimated_weights,
                                                             [self.n_nodes, self.n_nodes])
            _, loss, acc = self.get_prediction(features, self.edge_index, label, mask, reformed_adj_model)
        return loss, acc

    def get_train_edge(self, edge_index, features, n_p, idx_train):
        if n_p == 0: return None
        poten_edges = []
        train_idx_list = idx_train if not isinstance(idx_train, torch.Tensor) else idx_train.cpu().numpy()
        
        # Memory-Efficient similarity search for large datasets
        if len(features) > 20000:
            print(f" [NRGNN] Large dataset detected ({len(features)} nodes). Using sampled similarity search.")
            # Sample a subset of nodes to check similarity against train
            node_sample = np.random.choice(len(features), min(10000, len(features)), replace=False)
            for i in node_sample:
                sim = torch.div(torch.matmul(features[i], features[train_idx_list].T),
                                (features[i].norm() * features[train_idx_list].norm(dim=1)).clamp(min=1e-12))
                _, rank = sim.topk(min(n_p, len(train_idx_list)))
                indices = set(train_idx_list[rank.cpu().numpy()])
                indices = indices - set(edge_index[1, edge_index[0] == i].cpu().numpy())
                for j in indices:
                    poten_edges.append([i, j])
        else:
            for i in range(len(features)):
                sim = torch.div(torch.matmul(features[i], features[train_idx_list].T),
                                (features[i].norm() * features[train_idx_list].norm(dim=1)).clamp(min=1e-12))
                _, rank = sim.topk(min(n_p, len(train_idx_list)))
                indices = set(train_idx_list[rank.cpu().numpy()])
                indices = indices - set(edge_index[1, edge_index[0] == i].cpu().numpy())
                for j in indices:
                    poten_edges.append([i, j])
        
        if len(poten_edges) == 0: return None
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges, len(features)).to(self.device)
        return poten_edges

    def get_model_edge(self, pred):
        idx_add = self.idx_unlabel[(pred.max(dim=1)[0][self.idx_unlabel] > self.conf.model['p_u'])]
        
        # Memory-Efficient Edge Generation for large datasets
        n_unlabel = len(self.idx_unlabel)
        n_add = len(idx_add)
        
        if n_unlabel * n_add > 2000000: # Limit to 2M edges to save memory
            print(f" [NRGNN] Too many candidate edges ({n_unlabel}x{n_add}). Sampling 2M edges.")
            # Randomly sample pairs from the Cartesian product
            # This is much cheaper than creating the full product and then sampling
            sample_size = 2000000
            rand_unlabel_idx = torch.randint(0, n_unlabel, (sample_size,), device=self.device)
            rand_add_idx = torch.randint(0, n_add, (sample_size,), device=self.device)
            
            row = self.idx_unlabel[rand_unlabel_idx]
            col = idx_add[rand_add_idx]
            
            mask = (row != col)
            unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        else:
            row = self.idx_unlabel.repeat(len(idx_add))
            col = idx_add.repeat(len(self.idx_unlabel), 1).T.flatten()
            mask = (row != col)
            unlabel_edge_index = torch.stack([row[mask], col[mask]], dim=0)
            
        return unlabel_edge_index, idx_add
