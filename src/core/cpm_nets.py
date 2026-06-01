import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# CODE ADAPTED FROM: https://github.com/tbh-98/Reproducing-of-CPM-Nets-Cross-Partial-Multi-View-Networks

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    a = np.random.uniform(low,high,(fan_in,fan_out))
    a = a.astype('float32')
    a = torch.from_numpy(a)
    return a

class CPMNets(nn.Module):
    def __init__(self, layer_size, lsd_dim):
        """
        layer_size: node of each net
        lsd_dim: latent space dimensionality
        """
        super(CPMNets, self).__init__()
        layers = []
        in_dim = lsd_dim
        for out_dim in layer_size[:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, layer_size[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        out = self.net(h)
        # Added L2 norm
        return F.normalize(out, p=2, dim=1)

class CPMNet_Works(nn.Module):
    def __init__(self, device, view_dims, trainLen, testLen, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=0.1):
        super(CPMNet_Works, self).__init__()
        self.view_num = len(view_dims)
        self.view_dims = view_dims
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.device = device
        
        self.h_train = nn.Parameter(xavier_init(self.trainLen, self.lsd_dim).to(device))
        self.h_test = nn.Parameter(xavier_init(self.testLen, self.lsd_dim).to(device))
        
        self.net, self.train_net_op = self.build_model()

    def build_model(self):
        net = nn.ModuleDict()
        train_net_op = []
        for v_num in range(self.view_num):
            layer_size = [512, self.view_dims[v_num]]
            net[str(v_num)] = CPMNets(layer_size, self.lsd_dim).to(self.device)
            train_net_op.append(torch.optim.Adam(net[str(v_num)].parameters(), lr=self.learning_rate[0]))
        return net, train_net_op

    def calculate(self, h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[str(v_num)] = self.net[str(v_num)](h)
        return h_views

    def reconstruction_loss(self, h, x, sn):
        loss = 0
        x_pred = self.calculate(h)
        for num in range(self.view_num):
            loss += (torch.pow(x_pred[str(num)] - x[str(num)], 2.0) * sn[str(num)]).sum()
        return loss

    def classification_loss(self, label_onehot, gt, h_temp):
        F_h_h = torch.mm(h_temp, h_temp.T)
        F_hn_hn = torch.eye(F_h_h.shape[0], F_h_h.shape[1], device=self.device)
        F_h_h = F_h_h - F_h_h * F_hn_hn
        label_num = label_onehot.sum(0, keepdim=True) + 1e-8
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        gt1 = torch.max(F_h_h_mean, axis=1)[1]
        gt_ = gt1.type(torch.IntTensor) + 1
        F_h_h_mean_max = torch.max(F_h_h_mean, axis=1, keepdim=False)[0]
        gt_ = gt_.to(self.device).reshape([gt_.shape[0], 1])
        theta = torch.ne(gt.reshape(-1, 1), gt_).float().to(self.device)
        F_h_hn_mean_ = F_h_h_mean * label_onehot
        F_h_hn_mean = F_h_hn_mean_.sum(axis=1)
        F_h_h_mean_max = F_h_h_mean_max.reshape([F_h_h_mean_max.shape[0], 1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0], 1])
        return (torch.nn.functional.relu(theta + F_h_h_mean_max - F_h_hn_mean)).sum()


    def train_model(self, dataloader, num_classes, epoch, step=[5, 5]):
        # Used dataloader instead of full-batch
        train_hn_op = torch.optim.Adam([self.h_train], lr=self.learning_rate[1])
        
        for iter in range(epoch):
            epoch_rec = 0
            for batch_idx, data_dict, labels, sn_dict in dataloader:
                
                for v in range(self.view_num):
                    data_dict[str(v)] = data_dict[f"view_{v}"].to(self.device)
                    sn_dict[str(v)] = sn_dict[str(v)].unsqueeze(1).to(self.device)
                
                gt = (labels + 1).to(self.device)
                label_onehot = F.one_hot(labels, num_classes=num_classes).float().to(self.device)
                
                # Optimize networks
                for _ in range(step[0]):
                    h_batch = self.h_train[batch_idx].detach()
                    Reconstruction_LOSS = self.reconstruction_loss(h_batch, data_dict, sn_dict)
                    for v_num in range(self.view_num):
                        self.train_net_op[v_num].zero_grad()
                    Reconstruction_LOSS.backward()
                    for v_num in range(self.view_num):
                        self.train_net_op[v_num].step()
                        
                # Optimize H
                for _ in range(step[1]):
                    h_batch = self.h_train[batch_idx]
                    loss1 = self.reconstruction_loss(h_batch, data_dict, sn_dict)
                    loss2 = self.lamb * self.classification_loss(label_onehot, gt, h_batch)
                    
                    train_hn_op.zero_grad()
                    (loss1 + loss2).backward()
                    train_hn_op.step()
                    
                epoch_rec += loss1.item()
            if (iter+1) % 5 == 0:
                print(f"CPM Train epoch: {iter+1} | Rec loss: {epoch_rec/len(dataloader):.4f}")

    def test_model(self, dataloader, epoch, step=5):
        adj_hn_op = torch.optim.Adam([self.h_test], lr=self.learning_rate[1])
        for v_num in range(self.view_num):
            self.net[str(v_num)].eval()
            
        for iter in range(epoch):
            epoch_rec = 0
            for batch_idx, data_dict, _, sn_dict in dataloader:
                for v in range(self.view_num):
                    data_dict[str(v)] = data_dict[f"view_{v}"].to(self.device)
                    sn_dict[str(v)] = sn_dict[str(v)].unsqueeze(1).to(self.device)
                    
                for _ in range(step):
                    h_batch = self.h_test[batch_idx]
                    Reconstruction_LOSS = self.reconstruction_loss(h_batch, data_dict, sn_dict)
                    
                    adj_hn_op.zero_grad()
                    Reconstruction_LOSS.backward()
                    adj_hn_op.step()
                epoch_rec += Reconstruction_LOSS.item()
            if (iter+1) % 5 == 0:
                print(f"CPM Test epoch: {iter+1} | Rec loss: {epoch_rec/len(dataloader):.4f}")
