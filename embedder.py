import torch
from utils import process
import scipy as sp
from sklearn.model_selection import KFold
import scipy.io as scio
import numpy as np
import random
import sklearn
class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat()
            features = process.preprocess_features(features)
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "amazon":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_amazon(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "aminer":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_aminer(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "freebase":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_freebase(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "yelp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_yelp(args.sc)
            features = process.preprocess_features(features)
            args.ft_size = features[0].shape[1]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "slap":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_slap(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "ogbn-mag":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_mag(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "aminerlarge":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_aminer_large2(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
            data = []
            data.append(adj_list[0])
            data.append(adj_list[1])
            data.append(features)
            data.append(labels)
        if args.dataset == "graphcs":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_graphcs(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "freebaselarge":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_freebase_HGB()
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
        if args.dataset == "aminermiddle":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_aminermiddle(args.sc)
            args.ft_size = features.shape[1]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
            adj_list = [adj.to_dense() for adj in adj_list]
            adj_list = [process.normalize_graph(adj) for adj in adj_list]
            adj_list = [adj.to_sparse() for adj in adj_list]



        if args.dataset in ["dblp", "acm", "imdb", "amazon", "freebase"]:
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
            adj_list = [adj.to_dense() for adj in adj_list]
            adj_list = [process.normalize_graph(adj) for adj in adj_list]
            if args.sparse_adj:
                adj_list = [adj.to_sparse() for adj in adj_list]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
            args.ft_size = features.shape[1]
            adj_fusion = process.sparse_mx_to_torch_sparse_tensor(adj_fusion)
            adj_fusion = adj_fusion.to_dense()
            adj_fusion = process.normalize_graph(adj_fusion)
            adj_fusion = adj_fusion.to_sparse()
        self.adj_list = adj_list
        # self.data = data
        # self.adj_fusion = adj_fusion
        # adj_list = [process.normalize_adj(adj) for adj in adj_list]
        # self.adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).to(args.device)
        self.idx_train = torch.LongTensor(idx_train).to(args.device)
        self.idx_val = torch.LongTensor(idx_val).to(args.device)
        self.idx_test = torch.LongTensor(idx_test).to(args.device)

        self.args = args


