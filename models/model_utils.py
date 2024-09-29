import torch as t
from torch import nn
from torch.nn import init
import dgl.function as fn
from config.configurator import configs
import torch.nn.functional as F
import math
import scipy.sparse as sp
import numpy as np


class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (t.rand(edgeNum) + keep_rate).floor().type(t.bool)
        newVals = vals[mask]  # / keep_rate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class NodeDrop(nn.Module):
    def __init__(self):
        super(NodeDrop, self).__init__()

    def forward(self, embeds, keep_rate):
        if keep_rate == 1.0:
            return embeds
        data_config = configs['data']
        node_num = data_config['user_num'] + data_config['item_num']
        mask = (t.rand(node_num) + keep_rate).floor().view([-1, 1])
        return embeds * mask


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=False,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = "both"
        if weight:
            self.weight = nn.Parameter(t.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        graph = graph.local_var()

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)  # outdegree of nodes
            norm = t.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)  # (n, 1)
            norm = t.reshape(norm, shp)  # (n, 1)
            # feat = feat * norm

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = t.matmul(feat, weight)
            feat = feat * norm
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_u(u='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = t.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = t.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = t.reshape(norm, shp)
            rst = rst * norm
        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 activation):
        super(GCN, self).__init__()
        self.g = g
        self.layer = GraphConv(in_feats, n_hidden, weight=False, activation=activation)

    def forward(self, features):
        h = features
        h = self.layer(self.g, h)
        return h


def message_func(edges):
    return {'m': edges.src['n_f'] + edges.data['e_f']}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            init.xavier_uniform_(self.u_w)
            init.xavier_uniform_(self.v_w)
        self._activation = activation

    def forward(self, graph, u_f, v_f, e_f):
        with graph.local_scope():
            if self.weight:
                u_f = t.mm(u_f, self.u_w)
                v_f = t.mm(v_f, self.v_w)
            node_f = t.cat([u_f, v_f], dim=0)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.edata['e_f'] = e_f
            graph.update_all(message_func=message_func, reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.d_k = hidden_size // num_heads
        self.n_h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def _cal_attention(self, query, key, value, mask=None, dropout=None):
        scores = t.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return t.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
        #
        # return self.w_2(self.dropout(self.w_1(x)))


class Subtract_time(nn.Module):
    def __init__(self, position_emb, item_num, emb_size, max_len, decay_factor=0.5):
        """
        自己编写，处理时间序列
        """
        super().__init__()
        self.max_seq_len = max_len
        self.decay = decay_factor
        self.emb_size = emb_size
        self.position_embedding_item = position_emb  # 100*64
        self.position_emb = PositionalEncoding(self.emb_size, dropout=0.5, max_len=self.max_seq_len)
        self.dropout = nn.Dropout(p=0.1)
        self.alpha = 0.5


    def forward(self, item_seq_emb, batch_seqs_item, batch_last_time):
        tar_time_mat = t.zeros_like(batch_seqs_item) + batch_last_time.unsqueeze(1)
        subtract_time_mat =  tar_time_mat - batch_seqs_item
        position_ids_item = t.arange(batch_seqs_item[0].size(0), dtype=t.long, device=batch_seqs_item.device)
        position_ids_items = position_ids_item.unsqueeze(0).expand_as(batch_seqs_item)  # .expand_as扩展维度
        # # position_embedding_item = self.position_embedding(position_ids_item)
        position_embedding_item = self.position_embedding_item(position_ids_items)
        pos_emb = self.position_emb(position_embedding_item)
        #  方式1：先进行线性变化，在进行哈德曼积
        # S1 = self.w_1(subtract_time_mat)
        # S2 = self.w_2(subtract_time_mat)
        # S = S1*S2
        # out_mat = 初始权重值 * torch.exp(-S / self.decay)
        #  方式2:进行衰减函数公式上的变换
        out_mat = t.tensor(1 / (math.e + self.alpha * (abs(subtract_time_mat) / (24 * 60 * 60))))

        #  此处是指数衰减函数======================================================================================
        # out_mat = t.tensor(t.exp(-0.1*(abs(subtract_time_mat))))
        #  此处是指数衰减函数======================================================================================
        #  此处是对数衰减函数======================================================================================
        # out_mat = 1 - t.log((abs(subtract_time_mat)+math.e) / (t.max(subtract_time_mat) - t.min(subtract_time_mat) + math.e))
        #  此处是对数衰减函数======================================================================================
        #  此处是线性衰减函数======================================================================================
        # out_mat = 1 - abs(subtract_time_mat)
        #  此处是线性衰减函数======================================================================================

        #  方式3：将subtract_time_mat直接进行线性变换再求衰减函数
        # S1 = self.w_1(subtract_time_mat)
        # out_mat = 初始权重值 * torch.exp(-S1 / self.decay)
        # 归一化处理
        # out_mat = self.activation(out_mat)
        # transf = self.w_1(out_mat)
        # feat = self.w(torch.multiply(transf, out_mat))
        # out = torch.cat((graph_emb, feat))
        # return torch.multiply(transf, out_mat)
        # out_mat = out_mat.unsqueeze(-1).cuda()  #  按时间步应用，按序列应用，注意力机制，逐元素加权
        # out = 0.1*(item_seq_emb * (out_mat.unsqueeze(-1).cuda())) + pos_emb + item_seq_emb
        out = item_seq_emb + (out_mat.unsqueeze(-1).cuda()) + pos_emb
        # out = torch.sum(out, dim=0)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=50):
        """
        位置编码器类的初始化函数

        共有三个参数，分别是
        d_model：词嵌入维度
        dropout: dropout触发比率
        max_len：每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围，
        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        from torch.autograd import Variable
        x = 0.01*Variable(self.pe[:, :x.size(1)], requires_grad=True)  # x +
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, feed_forward_size, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout_rate)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, d_ff=feed_forward_size,
                                                    dropout=dropout_rate)
        self.input_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.output_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, item_num, emb_size, max_len, position_emb, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.token_emb = nn.Embedding(item_num, emb_size, padding_idx=0)
        self.token_emb = nn.Embedding(item_num, emb_size)
        self.position_emb = position_emb
        self.position_emb_01 = PositionalEncoding(emb_size, dropout=0.5, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size
        self.max_len = max_len
        # self.encoder = nn.ModuleList()  # 用于储存不同module的容器
        # for i in range(0, 1):
        #     self.encoder.append(GCN_layer())
        # self.subtract_time = Subtract_time(self.position_emb, item_num + 2, emb_size, max_len)

    def position_encoding_01(self, pos, d_model):
        pe1 = t.zeros(self.max_len, d_model)
        pe2 = t.zeros(self.max_len, d_model)
        angle_rates = 1/t.exp(t.arange(0, d_model) *
                         -(math.log(10000.0) / d_model))
        pe1[:, :] = t.sin(pos * angle_rates)
        pe2[:, :] = t.cos(pos * angle_rates)
        return pe1.cuda(), pe2.cuda()

    def generate_negative_examples(self, original_embedding, num_examples):
        neg_examples1 = original_embedding.clone()
        neg_examples2 = original_embedding.clone()

        # Choose random positions to modify
        positions_to_modify = t.randint(0, original_embedding.size(1), (num_examples,), dtype=t.long)


            # Generate position encoding for the selected position
        pe1, pe2 =self.position_encoding_01(t.range(1,original_embedding.shape[1]).unsqueeze(1),original_embedding.shape[2],)
                # [self.position_encoding_01(pos.item(), i, original_embedding.size(2)) for i in
                #  range(original_embedding.size(2))], device=original_embedding.device)

            # Modify embedding values for both negative examples using position encoding
        neg_examples1 += pe1
        neg_examples2 += pe2

        return pe1, pe2
    def forward(self, batch_seqs):
        batch_size = batch_seqs.size(0)


        pos_emb = self.position_emb.weight.unsqueeze(
            0).repeat(batch_size, 1, 1)
        pos_emb_01 = self.position_emb_01(pos_emb).repeat(batch_size, 1, 1)
        x = self.token_emb(batch_seqs) + pos_emb
        # original_embedding = x
        # num_negative_examples = 10
        # neg_examples1, neg_examples2 = self.generate_negative_examples(original_embedding, num_negative_examples)
        # x1 = neg_examples1
        # x2 = neg_examples2


        x1 = self.token_emb(batch_seqs)
        x2 = self.token_emb(batch_seqs) + pos_emb_01


        # x = self.subtract_time(x, batch_seqs_time, batch_last_time) + pos_emb_01

        return self.dropout(x), self.dropout(x1), self.dropout(x2)


class MultiHeadAttention1(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.d_k = hidden_size // num_heads
        self.n_h = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_linear = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def _cal_attention(self, query, key, value, mask=None, dropout=None):
        import math
        scores = t.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return t.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k  从d_model => h x d_k中批量进行所有线性投影
        query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.  对所有投射向量进行批量关注。
        x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

        return self.output_linear(x).squeeze(1)

class GCN_layer(nn.Module):
    def __init__(self):
        super(GCN_layer, self).__init__()
        self.temp_mha = MultiHeadAttention1(8, 64)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if type(sparse_mx) != sp.coo_matrix:
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = t.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = t.from_numpy(sparse_mx.data).float()
        shape = t.Size(sparse_mx.shape)
        return t.sparse.FloatTensor(indices, values, shape)
    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (d_mat_inv_sqrt).dot(adj).dot(d_mat_inv_sqrt).tocoo()

    def forward(self, features, Mat, index):
        subset_Mat = Mat
        subset_features = features
        subset_Mat = self.normalize_adj(subset_Mat)
        subset_sparse_tensor = self.sparse_mx_to_torch_sparse_tensor(subset_Mat).cuda()
        out_features = t.spmm(subset_sparse_tensor, subset_features)
        new_features = t.empty(features.shape).cuda()
        new_features[index] = out_features
        dif_index = np.setdiff1d(t.arange(features.shape[0]), index)
        new_features[dif_index] = features[dif_index]
        new_features = self.temp_mha(new_features, new_features, new_features, mask=None)
        return new_features


class DGIEncoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, activation):
        super(DGIEncoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, activation)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = t.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class DGIDiscriminator(nn.Module):
    def __init__(self, n_hidden):
        super(DGIDiscriminator, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(t.empty(n_hidden, n_hidden)))
        self.loss = nn.BCEWithLogitsLoss(reduction='none')  # combines a Sigmoid layer and the BCELoss

    def forward(self, node_embedding, graph_embedding, corrupt=False):
        score = t.sum(node_embedding * graph_embedding, dim=1)

        if corrupt:
            res = self.loss(score, t.zeros_like(score))
        else:
            res = self.loss(score, t.ones_like(score))
        return res
