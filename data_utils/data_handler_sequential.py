import pickle
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_sequential import SequentialDataset
import torch as t
import torch.utils.data as data
from os import path
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix


class DataHandlerSequential:
    def __init__(self):
        if configs['data']['name'] == 'ml-20m':
            predir = './datasets/sequential/ml-20m_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'sports':
            predir = './datasets/sequential/sports_seq/'
            configs['data']['dir'] = predir
        else:
            predir = './datasets/sequential/' + configs['data']['name'] +'/'
            configs['data']['dir'] = predir

            
        self.trn_file = path.join(predir, 'train.csv')
        self.item_graph = path.join(predir, 'cd_item_trn')
        self.val_file = path.join(predir, 'test.csv')
        self.tst_file = path.join(predir, 'test.csv')
        self.max_item_id = 0

    def _read_tsv_to_user_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": [], "item_time_seq":[], "item_time_id":[]}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            # skip header
            line = f.readline()
            while line:
                uid, seq, last_item, time1, time2 = line.strip().split('\t')
                seq = seq.split(' ')
                time1= time1.split(' ')
                # time2 = time2.split(' ')
                seq = [int(item) for item in seq]
                time1 = [float(item) for item in time1]
                # time2 = [float(item) for item in time2]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))
                user_seqs["item_time_seq"].append(time1)
                user_seqs["item_time_id"].append(float(time2))

                self.max_item_id = max(
                    self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs

    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        # item originally starts with 1
        configs['data']['item_num'] = self.max_item_id

    def _seq_aug(self, user_seqs):
        user_seqs_aug = {"uid": [], "item_seq": [], "item_id": [], "item_time_seq":[], "item_time_id":[]}
        for uid, seq, last_item, time1, time2 in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"], user_seqs["item_time_seq"], user_seqs["item_time_id"]):
            user_seqs_aug["uid"].append(uid)
            user_seqs_aug["item_seq"].append(seq)
            user_seqs_aug["item_id"].append(last_item)
            user_seqs_aug["item_time_seq"].append(time1)
            user_seqs_aug["item_time_id"].append(time2)

            for i in range(1, len(seq)-1):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq[:i])
                user_seqs_aug["item_id"].append(seq[i])
                user_seqs_aug["item_time_seq"].append(time1[:i])
                user_seqs_aug["item_time_id"].append(time1[i])

        return user_seqs_aug

    def make_torch_adj(self, mat):
        mat = (mat + sp.eye(mat.shape[0]))  # 对角线加1，自身交互不为0   54756*2=109512    项目项目交互
        mat = (mat != 0) * 1.0
        mat = self.normalize(mat)  # 正则化
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return sp.csr_matrix((vals, mat.row, mat.col), shape=shape)

    def normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def make_all_one_adj(self, adj):
        idxs = adj._indices()
        vals = t.ones_like(adj._values())
        shape = adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def _read_seq(self, item_graph):
        with open(item_graph, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != csr_matrix:
            ret = sp.csr_matrix(ret)
        ii_dok = ret
        # ii_adj = self.make_torch_adj(ret)  # 交互正则化处理   nnz=1793756
        # ii_adj_all_one = self.make_all_one_adj(ii_adj)  # 交互归1处理  nnz=1793756
        return ii_dok, 0, 0

    def load_data(self):
        user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
        user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
        self.ii_dok, self.ii_adj, self.ii_adj_all_one = self._read_seq(self.item_graph)
        self._set_statistics(user_seqs_train, user_seqs_test)

        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            user_seqs_aug = self._seq_aug(user_seqs_train)
            trn_data = SequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug)
        else:
            trn_data = SequentialDataset(user_seqs_train)
        tst_data = SequentialDataset(user_seqs_test, mode='test')
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
