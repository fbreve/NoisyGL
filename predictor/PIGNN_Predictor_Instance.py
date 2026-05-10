from predictor.Base_Predictor import Predictor
from predictor.module.PIGNN import GCN_pignn
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
import nni
import torch_geometric.utils as utils


class pignn_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        super().__init__(conf, data, device)

    def method_init(self, conf, data):
        self.model = GCN_pignn(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                               out_channels=conf.model['n_classes'],
                               n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                               norm_info=conf.model['norm_info'],
                               act=conf.model['act'], input_layer=conf.model['input_layer'],
                               output_layer=conf.model['output_layer']).to(self.device)
        self.model_mi = GCN_pignn(in_channels=conf.model['n_feat'], hidden_channels=conf.model['n_hidden'],
                                  out_channels=conf.model['n_classes'],
                                  n_layers=conf.model['n_layer'], dropout=conf.model['dropout'],
                                  norm_info=conf.model['norm_info'],
                                  act=conf.model['act'], input_layer=conf.model['input_layer'],
                                  output_layer=conf.model['output_layer']).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])
        self.optim_mi = torch.optim.Adam(
            list(self.model_mi.parameters()),
            lr=self.conf.training['lr'],
            weight_decay=self.conf.training['weight_decay'])

    def get_prediction(self, features, adj, label=None, mask=None):
        output, output_product = self.model(features, adj)
        loss, acc = None, None
        if (label is not None) and (mask is not None):
            loss = F.nll_loss(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return output, loss, acc

    def train(self):
        # Determine if we need sampling based on dataset size
        # Reduced to 5000 to avoid giant dense matrices (722MB+) on 8GB GPUs
        use_sampling = self.n_nodes > 5000
        if use_sampling:
            print(f" [PIGNN] Large dataset ({self.n_nodes} nodes). Using edge sampling for context loss.")

        # Create sparse adjacency tensor on GPU
        t_gpu_setup = time.time()
        try:
            adj_gpu = torch.sparse_coo_tensor(
                self.edge_index, 
                self.edge_weight if self.edge_weight is not None else torch.ones(self.edge_index.shape[1], device=self.device),
                [self.n_nodes, self.n_nodes]
            ).coalesce().to_sparse_csr()
        except:
            adj_gpu = torch.sparse_coo_tensor(
                self.edge_index, 
                self.edge_weight if self.edge_weight is not None else torch.ones(self.edge_index.shape[1], device=self.device),
                [self.n_nodes, self.n_nodes]
            ).coalesce()
        self._t_gpu += (time.time() - t_gpu_setup)

        data_context_size = self.edge_index.shape[1]
        pos_weight = torch.tensor(
            [float(self.n_nodes * self.n_nodes - data_context_size) / data_context_size],
            device=self.device
        )

        for epoch in range(self.conf.training['n_epochs']):
            improve = ''
            t0 = time.time()
            self.model.train()
            self.model_mi.train()
            self.optim.zero_grad()
            self.optim_mi.zero_grad()
            
            features = self.feats
            
            if use_sampling:
                # ── Optimized Sampled Context Loss ──
                t_cpu_sampling = time.time()
                num_samples = 1000000 # 1M samples
                pos_idx = torch.randint(0, data_context_size, (num_samples // 2,), device=self.device)
                pos_pairs = self.edge_index[:, pos_idx]
                neg_pairs = utils.negative_sampling(self.edge_index, num_nodes=self.n_nodes, num_neg_samples=num_samples // 2)
                self._t_cpu += (time.time() - t_cpu_sampling)
                
                t_gpu_start = time.time()
                sample_pairs = torch.cat([pos_pairs, neg_pairs], dim=1)
                sample_labels = torch.cat([torch.ones(pos_pairs.shape[1], device=self.device), 
                                           torch.zeros(neg_pairs.shape[1], device=self.device)])
                
                # Forward MI model
                out_mi, _ = self.model_mi(features, adj_gpu)
                logits_mi = torch.sum(out_mi[sample_pairs[0]] * out_mi[sample_pairs[1]], dim=1)
                
                norm = self.n_nodes * self.n_nodes / float(num_samples)
                loss_mi = norm * F.binary_cross_entropy_with_logits(logits_mi, sample_labels, pos_weight=pos_weight)
                loss_mi.backward()
                self.optim_mi.step()
                
                # Forward Main model
                out, _ = self.model(features, adj_gpu)
                logits = torch.sum(out[sample_pairs[0]] * out[sample_pairs[1]], dim=1)
                
                # Mask calculation (detached)
                with torch.no_grad():
                    mask_sampled = torch.sigmoid(logits_mi)
                    mask_sampled[sample_labels == 0] = 1 - mask_sampled[sample_labels == 0]

                loss_train = F.nll_loss(F.log_softmax(out[self.train_mask], dim=1), self.noisy_label[self.train_mask])
                
                if epoch > self.conf.training['start_epoch']:
                    loss_context = norm * (F.binary_cross_entropy_with_logits(
                        logits, sample_labels, pos_weight=pos_weight, reduction='none') * mask_sampled).mean()
                else:
                    loss_context = norm * F.binary_cross_entropy_with_logits(logits, sample_labels, pos_weight=pos_weight)
                
                loss_train += loss_context
                self._t_gpu += (time.time() - t_gpu_start)
                
            else:
                # ── Original Dense Context Loss (for small datasets) ──
                t_gpu_dense = time.time()
                labels_context = adj_gpu.to_dense() + torch.eye(self.n_nodes, device=self.device)
                out_mi, out_product_mi = self.model_mi(features, adj_gpu)
                norm = self.n_nodes * self.n_nodes / float((self.n_nodes * self.n_nodes - data_context_size) * 2)
                loss_mi = norm * F.binary_cross_entropy_with_logits(out_product_mi, labels_context, pos_weight=pos_weight)
                loss_mi.backward()
                self.optim_mi.step()

                output, output_product = self.model(features, adj_gpu)
                with torch.no_grad():
                    mask = torch.zeros_like(output_product).view(-1).to(self.device)
                    pos_position = labels_context.view(-1).bool()
                    mask[pos_position] = torch.sigmoid(output_product).view(-1)[pos_position]
                    mask[~pos_position] = 1 - torch.sigmoid(output_product).view(-1)[~pos_position]
                    mask = mask.view(self.n_nodes, self.n_nodes)

                loss_train = F.nll_loss(output[self.train_mask], self.noisy_label[self.train_mask])
                if epoch > self.conf.training['start_epoch']:
                    loss_context = norm * (F.binary_cross_entropy_with_logits(
                        output_product, labels_context, pos_weight=pos_weight, reduction='none') * mask.detach()).mean()
                else:
                    loss_context = norm * F.binary_cross_entropy_with_logits(
                        output_product, labels_context, pos_weight=pos_weight)
                loss_train += loss_context
                self._t_gpu += (time.time() - t_gpu_dense)

            t_gpu_final = time.time()
            loss_train.backward()
            self.optim.step()

            # Evaluate
            loss_val, acc_val = self.evaluate_internal(adj_gpu, self.noisy_label, self.val_mask)
            self._t_gpu += (time.time() - t_gpu_final)
            
            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag:
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.best_acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                                 (out if use_sampling else output)[self.train_mask].detach().cpu().numpy())
                self.result['train'] = self.best_acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop:
                break

            if self.conf.training['debug']:
                acc_train = self.metric(self.noisy_label[self.train_mask].cpu().numpy(),
                                         (out if use_sampling else output)[self.train_mask].detach().cpu().numpy())
                print(
                    "Epoch {:05d} | Time(s) {:.4f} | Loss(train) {:.4f} | Acc(train) {:.4f} | Loss(val) {:.4f} | Acc(val) {:.4f} | {}".format(
                        epoch + 1, time.time() - t0, loss_train.item(), acc_train, loss_val, acc_val, improve), flush=True)

        t_gpu_test = time.time()
        loss_test, acc_test = self.test_internal(adj_gpu, self.test_mask)
        self._t_gpu += (time.time() - t_gpu_test)
        self.result['test'] = acc_test
        return self.result

    def evaluate_internal(self, adj_gpu, label, mask):
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(self.feats, adj_gpu)
            loss = F.nll_loss(output[mask], label[mask])
            acc = self.metric(label[mask].cpu().numpy(), output[mask].detach().cpu().numpy())
        return loss, acc

    def test_internal(self, adj_gpu, mask):
        if self.weights is not None:
            self.model.load_state_dict(self.weights)
        return self.evaluate_internal(adj_gpu, self.clean_label, mask)
