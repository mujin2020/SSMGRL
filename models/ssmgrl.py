import os
import torch.nn as nn
from tqdm import tqdm
from block import fusions
from evaluate import evaluate
from embedder import embedder
import numpy as np
import random as random
import torch
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)



class SSMGRL(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.cfg = args.cfg
        self.sigm = nn.Sigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#
        features = self.features.to(self.args.device)
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]
        print("Started training...")
        model = trainer(self.args)
        model = model.to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        model.train()

        for epoch in tqdm(range(self.args.nb_epochs+1)):
            optimiser.zero_grad()
            loss = model(features, adj_list)

            loss.backward()
            optimiser.step()
        # torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key))
        if self.args.use_pretrain:
            model.load_state_dict(torch.load('saved_model/best_{}_{}.pkl'.format(self.args.dataset,self.args.custom_key)))
        print('loss', loss)
        print("Evaluating...")
        model.eval()
        hf = model.embed(features, adj_list)
        macro_f1s, micro_f1s, k1, st = evaluate(hf, self.idx_train, self.idx_val, self.idx_test, self.labels,task=self.args.custom_key,epoch = self.args.test_epo,lr = self.args.test_lr,iterater=self.args.iterater) #,seed=seed
        return macro_f1s, micro_f1s, k1, st




def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cfg = args.cfg
        self.MLP1 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP2 = make_mlplayers(args.ft_size, args.cfg)
        self.MLP3 = make_mlplayers(args.ft_size, args.cfg)
        length = args.length
        self.w_list = nn.ModuleList([nn.Linear(cfg[-1], cfg[-1], bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(cfg[-1], 1) for _ in range(length)])
        self.W = nn.Parameter(torch.zeros(size=(length * cfg[-1], cfg[-1])))
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)




    def combine_att(self, h_list):
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h

    def forward(self, x, adj_list=None,epoch=0,encode_time=[]):
        x = F.dropout(x, self.args.dropout, training=self.training)
        if self.args.length == 2:
            h_a = self.MLP1(x)
            h_a_1 = self.MLP2(x)

        elif self.args.length == 3:

            h_a = self.MLP1(x)
            h_a_1 = self.MLP2(x)
            h_a_2 = self.MLP3(x)

        h_p_list = []
        i = 0
        for adj in adj_list:
            if self.args.sparse:
                if i == 0 :
                    h_p = torch.spmm(adj, h_a)
                elif i == 1:
                    h_p = torch.spmm(adj, h_a_1)
                elif i == 2:
                    h_p = torch.spmm(adj, h_a_2)
                h_p_list.append(h_p)
                # h_p_list_3.append(h_p_3)
            else:
                h_p = torch.mm(adj, h_a)
                h_p_list.append(h_p)
            i += 1
        if self.args.length == 2:
            c = (h_p_list[1]).T @ (h_p_list[0])
            c_0 = (h_p_list[0]).T @ (h_a)
            c_1 = (h_p_list[1]).T @ (h_a_1)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss_0 = on_diag + self.args.lambd0 * off_diag

            on_diag_0 = torch.diagonal(c_0).add_(-1).pow_(2).sum()
            off_diag_0 = off_diagonal(c_0).pow_(2).sum()
            loss_1 = on_diag_0 + self.args.lambd1 * off_diag_0
            #
            on_diag_1 = torch.diagonal(c_1).add_(-1).pow_(2).sum()
            off_diag_1 = off_diagonal(c_1).pow_(2).sum()
            loss_2 = on_diag_1 + self.args.lambd2 * off_diag_1
            loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2
        elif self.args.length == 3:
            c = (h_p_list[1]).T @ (h_p_list[0])
            c_0 = (h_p_list[1]).T @ (h_p_list[2])
            c_1 = (h_p_list[0]).T @ (h_p_list[2])
            c_2 = (h_p_list[0]).T @ (h_a)
            c_3 = (h_p_list[1]).T @ (h_a_1)
            c_4 = (h_p_list[1]).T @ (h_a_2)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss_0 = on_diag + self.args.lambd0 * off_diag

            on_diag_0 = torch.diagonal(c_0).add_(-1).pow_(2).sum()
            off_diag_0 = off_diagonal(c_0).pow_(2).sum()
            loss_1 = on_diag_0 + self.args.lambd1 * off_diag_0
            #
            on_diag_1 = torch.diagonal(c_1).add_(-1).pow_(2).sum()
            off_diag_1 = off_diagonal(c_1).pow_(2).sum()
            loss_2 = on_diag_1 + self.args.lambd2 * off_diag_1

            on_diag_2 = torch.diagonal(c_2).add_(-1).pow_(2).sum()
            off_diag_2 = off_diagonal(c_2).pow_(2).sum()
            loss_3 = on_diag_2 + self.args.lambd3 * off_diag_2

            on_diag_3 = torch.diagonal(c_3).add_(-1).pow_(2).sum()
            off_diag_3 = off_diagonal(c_3).pow_(2).sum()
            loss_4 = on_diag_3 + self.args.lambd4 * off_diag_3

            on_diag_4 = torch.diagonal(c_4).add_(-1).pow_(2).sum()
            off_diag_4 = off_diagonal(c_4).pow_(2).sum()
            loss_5 = on_diag_4 + self.args.lambd5 * off_diag_4
            loss = loss_0 + self.args.w_loss1 * loss_1 + self.args.w_loss2 * loss_2\
                                + self.args.w_loss3 * loss_3 + self.args.w_loss4 * loss_4 + self.args.w_loss5 * loss_5
        return loss

    def embed(self, x, adj_list=None,adj_fusion=None):
        if self.args.length == 2:
            h_p = self.MLP1(x)
            h_p_1 = self.MLP2(x)
        elif self.args.length == 3:
            h_p = self.MLP1(x)
            h_p_1 = self.MLP2(x)
            h_p_2 = self.MLP3(x)

        h_p_list = []
        i =0
        for adj in adj_list:
            if self.args.sparse:
                if i == 0:
                    h_p = torch.spmm(adj, h_p)
                if i == 1:
                    h_p = torch.spmm(adj, h_p_1)
                if i == 2:
                    h_p = torch.spmm(adj, h_p_2)
                h_p_list.append(h_p)
            else:
                h_p = torch.mm(adj, h_p)
                h_p_list.append(h_p)
            i += 1
        h_fusion = self.combine_att(h_p_list)

        return  h_fusion.detach()




def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None):
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
            # result = nn.Sequential(*layers)
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)#, result

