# import math
# import random
# from models.base_model import BaseModel
# from models.model_utils import TransformerLayer, TransformerEmbedding, Subtract_time, PositionalEncoding
# import numpy as np
# import torch
# from torch import nn
# from config.configurator import configs
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class our(BaseModel):
#     def __init__(self, data_handler):
#         super(our, self).__init__(data_handler)
#         self.item_num = configs['data']['item_num']
#         self.emb_size = configs['model']['embedding_size']
#         self.max_len = configs['model']['max_seq_len']
#         self.mask_token = self.item_num + 1
#         # load parameters info
#         self.n_layers = configs['model']['n_layers']
#         self.n_heads = configs['model']['n_heads']
#         self.emb_size = configs['model']['embedding_size']
#         # the dimensionality in feed-forward layer
#         self.inner_size = 4 * self.emb_size
#         self.dropout_rate = configs['model']['dropout_rate']
#
#         self.batch_size = configs['train']['batch_size']
#         self.lmd = configs['model']['lmd']
#         self.tau = configs['model']['tau']
#         self.deca_factor = configs['model']['deca_factor']
#
#         self.position_emb = nn.Embedding(self.max_len, self.emb_size)
#         # self.position_emb_01 = PositionalEncoding(self.emb_size, dropout=0.5, max_len=self.max_len)
#
#         self.emb_layer = TransformerEmbedding(
#             self.item_num + 2, self.emb_size, self.max_len, self.position_emb)
#
#         self.transformer_layers = nn.ModuleList([TransformerLayer(
#             self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])
#
#         self.subtract_time = Subtract_time(self.position_emb, self.item_num + 2, self.emb_size, self.max_len,
#                                            self.deca_factor)
#
#         self.loss_func = nn.CrossEntropyLoss()
#
#         self.mask_default = self.mask_correlated_samples(
#             batch_size=self.batch_size)
#         self.cl_loss_func = nn.CrossEntropyLoss()
#
#         # parameters initialization
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
#
#     def _cl4srec_aug(self, batch_seqs):
#         def item_crop(seq, length, eta=0.6):
#             num_left = math.floor(length * eta)
#             crop_begin = random.randint(0, length - num_left)
#             croped_item_seq = np.zeros_like(seq)
#             if crop_begin != 0:
#                 croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):-crop_begin]
#             else:
#                 croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):]
#             return croped_item_seq.tolist(), num_left
#
#         def item_mask(seq, length, gamma=0.3):
#             num_mask = math.floor(length * gamma)
#             mask_index = random.sample(range(length), k=num_mask)
#             masked_item_seq = seq[:]
#             # token 0 has been used for semantic masking
#             mask_index = [-i-1 for i in mask_index]
#             masked_item_seq[mask_index] = self.mask_token
#             return masked_item_seq.tolist(), length
#
#         def item_reorder(seq, length, beta=0.6):
#             num_reorder = math.floor(length * beta)
#             reorder_begin = random.randint(0, length - num_reorder)
#             reordered_item_seq = seq[:]
#             shuffle_index = list(
#                 range(reorder_begin, reorder_begin + num_reorder))
#             random.shuffle(shuffle_index)
#             shuffle_index = [-i for i in shuffle_index]
#             reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
#             return reordered_item_seq.tolist(), length
#
#         # def item_time_disruption(seq, length, beta=0.1):
#         #     disruption_item_seq = seq[:]
#         #     # disruption = self.position_emb_01(disruption_item_seq)
#         #     # disruption_item_seq = disruption_item_seq + disruption
#         #     return disruption_item_seq.tolist(), length
#
#
#         seqs = batch_seqs.tolist()
#         lengths = batch_seqs.count_nonzero(dim=1).tolist()
#
#         aug_seq1 = []
#         aug_len1 = []
#         aug_seq2 = []
#         aug_len2 = []
#         for seq, length in zip(seqs, lengths):
#             seq = np.asarray(seq.copy(), dtype=np.int64)
#             if length > 1:
#                 switch = random.sample(range(3), k=2)
#             else:
#                 switch = [3, 3]
#                 aug_seq = seq
#                 aug_len = length
#             if switch[0] == 0:
#                 aug_seq, aug_len = item_crop(seq, length)
#             elif switch[0] == 1:
#                 aug_seq, aug_len = item_mask(seq, length)
#             elif switch[0] == 2:
#                 aug_seq, aug_len = item_reorder(seq, length)
#             # elif switch[0] == 3:
#             #     aug_seq, aug_len = item_time_disruption(seq, length)
#
#             if aug_len > 0:
#                 aug_seq1.append(aug_seq)
#                 aug_len1.append(aug_len)
#             else:
#                 aug_seq1.append(seq.tolist())
#                 aug_len1.append(length)
#
#             if switch[1] == 0:
#                 aug_seq, aug_len = item_crop(seq, length)
#             elif switch[1] == 1:
#                 aug_seq, aug_len = item_mask(seq, length)
#             elif switch[1] == 2:
#                 aug_seq, aug_len = item_reorder(seq, length)
#
#             if aug_len > 0:
#                 aug_seq2.append(aug_seq)
#                 aug_len2.append(aug_len)
#             else:
#                 aug_seq2.append(seq.tolist())
#                 aug_len2.append(length)
#
#         aug_seq1 = torch.tensor(
#             aug_seq1, dtype=torch.long, device=batch_seqs.device)
#         aug_seq2 = torch.tensor(
#             aug_seq2, dtype=torch.long, device=batch_seqs.device)
#         return aug_seq1, aug_seq2
#
#     def mask_correlated_samples(self, batch_size):
#         N = 2 * batch_size
#         mask = torch.ones((N, N), dtype=bool)
#         mask = mask.fill_diagonal_(0)
#         for i in range(batch_size):
#             mask[i, batch_size + i] = 0
#             mask[batch_size + i, i] = 0
#         return mask
#
#     def info_nce(self, z_i, z_j, temp, batch_size):
#         N = 2 * batch_size
#
#         z = torch.cat((z_i, z_j), dim=0)
#
#         sim = torch.mm(z, z.T) / temp
#
#         sim_i_j = torch.diag(sim, batch_size)
#         sim_j_i = torch.diag(sim, -batch_size)
#
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         if batch_size != self.batch_size:
#             mask = self.mask_correlated_samples(batch_size)
#         else:
#             mask = self.mask_default
#         negative_samples = sim[mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_samples.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         info_nce_loss = self.cl_loss_func(logits, labels)
#         return info_nce_loss
#
#     def forward(self, batch_seqs, batch_seqs_time, batch_last_time):
#         mask = (batch_seqs > 0).unsqueeze(1).repeat(
#             1, batch_seqs.size(1), 1).unsqueeze(1)
#         x,_,_ = self.emb_layer(batch_seqs, batch_seqs_time, batch_last_time)
#         y = (0.6* self.subtract_time(x, batch_seqs_time, batch_last_time) + 0.4*x) / 2
#
#         for transformer in self.transformer_layers:
#             x = transformer(y, mask)
#
#         output = x[:, -1, :]
#
#         return output # [B H]
#
#     # def item_time_disruption(self, batch_seqs, batch_seqs_time,batch_last_time):
#     #     mask = (batch_seqs > 0).unsqueeze(1).repeat(
#     #         1, batch_seqs.size(1), 1).unsqueeze(1)
#     #     disruption_item_seq = batch_seqs[:]
#     #     disruption_item_seq_emb,_,_ = self.emb_layer(disruption_item_seq, batch_seqs_time,batch_last_time)
#     #     disruption = self.position_emb_01(disruption_item_seq_emb)
#     #     disruption_item_seq = disruption_item_seq_emb + disruption
#     #     for transformer in self.transformer_layers:
#     #         disruption_item_seq = transformer(disruption_item_seq, mask)
#     #     disruption_item_seq = disruption_item_seq[:, -1, :]
#     #     return disruption_item_seq
#
#     def cal_loss(self, batch_data):
#         batch_user, batch_seqs, batch_last_items, batch_seqs_time,batch_last_time = batch_data
#         seq_output = self.forward(batch_seqs, batch_seqs_time, batch_last_time)
#
#         test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
#         logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
#         loss = self.loss_func(logits, batch_last_items)
#
#         # NCE
#         aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs)
#         seq_output1 = self.forward(aug_seq1, batch_seqs_time, batch_last_time)
#         seq_output2 = self.forward(aug_seq2, batch_seqs_time, batch_last_time)
#
#         # disruption_item_seq_emb = self.emb_layer(batch_seqs, batch_seqs_time,batch_last_time)
#         # seq_output3 = self.item_time_disruption(batch_seqs, batch_seqs_time,batch_last_time)
#
#         cl_loss = self.lmd * self.info_nce(
#             seq_output1, seq_output2, temp=self.tau, batch_size=aug_seq1.shape[0])
#
#         # cl_loss_1 = self.lmd * self.info_nce(
#         #     output1, output2, temp=self.tau, batch_size=aug_seq1.shape[0])
#
#         loss_dict = {
#             'rec_loss': loss.item(),
#             'cl_loss': cl_loss.item(),
#         }
#         return loss + cl_loss, loss_dict
#
#     def full_predict(self, batch_data):
#         batch_user, batch_seqs, batch_last_items, batch_seqs_time,batch_last_time  = batch_data
#         logits = self.forward(batch_seqs, batch_seqs_time, batch_last_time)
#         test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
#         scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
#         return scores




import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding, Subtract_time, PositionalEncoding, GCN_layer
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
from config.configurator import configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




class our(BaseModel):
    def __init__(self, data_handler):
        super(our, self).__init__(data_handler)
        self.item_num = configs['data']['item_num']
        self.num_users = configs['data']['user_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1
        # load parameters info
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        # the dimensionality in feed-forward layer
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']

        self.batch_size = configs['train']['batch_size']
        self.lmd = configs['model']['lmd']
        self.tau = configs['model']['tau']
        self.deca_factor = configs['model']['deca_factor']
        self.item_index = np.arange(0, self.item_num+1)


        self.position_emb = nn.Embedding(self.max_len, self.emb_size)
        self.position_emb_01 = PositionalEncoding(self.emb_size, dropout=0.5, max_len=self.max_len)
        self.subtract_time = Subtract_time(self.position_emb, self.item_num + 2, self.emb_size, self.max_len,
                                                   self.deca_factor)


        self.emb_layer = TransformerEmbedding(
            self.item_num + 2, self.emb_size, self.max_len, self.position_emb )

        # self.encoder = nn.ModuleList()  # 用于储存不同module的容器
        # for i in range(0, 2):
        #     self.encoder.append(GCN_layer())

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss()

        self.mask_default = self.mask_correlated_samples(
            batch_size=self.batch_size)
        self.cl_loss_func = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _cl4srec_aug(self, batch_seqs):
        def item_crop(seq, length, eta=0.6):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros_like(seq)
            if crop_begin != 0:
                croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):-crop_begin]
            else:
                croped_item_seq[-num_left:] = seq[-(crop_begin + num_left):]
            return croped_item_seq.tolist(), num_left

        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            # token 0 has been used for semantic masking
            mask_index = [-i-1 for i in mask_index]
            masked_item_seq[mask_index] = self.mask_token
            return masked_item_seq.tolist(), length

        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(
                range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            shuffle_index = [-i for i in shuffle_index]
            reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
            return reordered_item_seq.tolist(), length

        seqs = batch_seqs.tolist()
        lengths = batch_seqs.count_nonzero(dim=1).tolist()

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            seq = np.asarray(seq.copy(), dtype=np.int64)
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq1.append(aug_seq)
                aug_len1.append(aug_len)
            else:
                aug_seq1.append(seq.tolist())
                aug_len1.append(length)

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)

            if aug_len > 0:
                aug_seq2.append(aug_seq)
                aug_len2.append(aug_len)
            else:
                aug_seq2.append(seq.tolist())
                aug_len2.append(length)

        aug_seq1 = torch.tensor(
            aug_seq1, dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(
            aug_seq2, dtype=torch.long, device=batch_seqs.device)
        return aug_seq1, aug_seq2

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size):
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self.cl_loss_func(logits, labels)
        return info_nce_loss


    def item_time_disruption(self, batch_seqs, seq_output):

        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)

        # item_emb = self.token_emb.weight[:]
        # self.all_item_embeddings = [item_emb]

        # for i in range(len(self.encoder)):
        #     layer = self.encoder[i]
        #     if i == 0:
        #         self.itemEmbedding = layer(item_emb, ii_adj, self.item_index)
        #     else:
        #         self.itemEmbedding = layer(self.itemEmbedding, ii_adj, self.item_index)
        #     self.all_item_embeddings += [self.itemEmbedding]
        # batch_seqs_01 = self.all_item_embeddings[0]
        # batch_seqs_02 = self.all_item_embeddings[1]
        # batch_seqs_01 = batch_seqs_01[batch_seqs]
        # batch_seqs_02 = batch_seqs_02[batch_seqs]

        # position = torch.range(0,batch_seqs.shape[1])


        # disruption_item_seq_emb, _, _ = self.emb_layer(batch_seqs)
        original_embedding = seq_output
        num_negative_examples = 20
        neg_examples1, neg_examples2 = self.generate_negative_examples(original_embedding, num_negative_examples)
        # for transformer in self.transformer_layers:
        #     disruption_item_seq_emb_0 = transformer(neg_examples1, mask)
        #     disruption_item_seq_emb_1 = transformer(neg_examples2, mask)
        # disruption_item_seq_emb_0 = disruption_item_seq_emb_0[:, -1, :]
        # disruption_item_seq_emb_1 = disruption_item_seq_emb_1[:, -1, :]


        # disruption_item_seq = batch_seqs[:]
        # disruption_item_seq_emb_0, _, _ = self.emb_layer(batch_seqs_01)
        # disruption_item_seq_emb_1, _, _ = self.emb_layer(batch_seqs_02)
        # disruption_pos1 = self.position_emb_01(disruption_item_seq_emb_0)
        # disruption_pos2 = self.position_emb_01(disruption_item_seq_emb_1)
        # disruption_item_seq_0 = disruption_item_seq_emb_0 + disruption_pos1
        # disruption_item_seq_1 = disruption_item_seq_emb_1 + disruption_pos2
        # for transformer in self.transformer_layers:
        #     disruption_item_seq_emb_0 = transformer(disruption_item_seq_0, mask)
        #     disruption_item_seq_emb_1 = transformer(disruption_item_seq_1, mask)
        # disruption_item_seq_emb_0 = disruption_item_seq_emb_0[:, -1, :]
        # disruption_item_seq_emb_1 = disruption_item_seq_emb_1[:, -1, :]
        return neg_examples1, neg_examples2

    def position_encoding_01(self, pos, i, d):
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d, dtype=torch.float32))
        return pos * angle_rates

    def generate_negative_examples(self, original_data, num_examples):
        neg_examples1 = original_data.clone()
        neg_examples2 = original_data.clone()

        # Choose random positions to modify
        positions_to_modify = torch.randint(0, original_data.size(1), (num_examples,), dtype=torch.long)

        for pos in positions_to_modify:
            # Generate position encoding for the selected position
            pos_encoding = torch.tensor(
                [self.position_encoding_01(pos.item(), i, original_data.size(-1)) for i in range(original_data.size(-1))], device=original_data.device)

            # Modify data values for both negative examples using element-wise addition
            neg_examples1[pos,:] += 0.1*torch.sin(pos_encoding)
            neg_examples2[pos,:] += 0.1*torch.cos(pos_encoding)

        return neg_examples1, neg_examples2

    def forward(self, batch_seqs, batch_seqs_time, batch_last_time):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1)
        x,x1,x2 = self.emb_layer(batch_seqs)
        y =  self.subtract_time(x, batch_seqs_time, batch_last_time)
        # y1 = self.subtract_time(x1, batch_seqs_time, batch_last_time)
        # y2 = self.subtract_time(x2, batch_seqs_time, batch_last_time)
        # pycharm:x  不加pos , vs2:x2  +pos ,vs1:x  +pos  vs3:  x1 +pos
        for transformer in self.transformer_layers:
            x = transformer(y, mask)
            # x1 = transformer(y1, mask)
            # x2 = transformer(y2, mask)
        output = x[:, -1, :]
        # output1 = x1[:, -1, :]
        # output2 = x2[:, -1, :]
        return output   #, output1, output2  # [B H]

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items, batch_seqs_time, batch_last_time = batch_data
        seq_output = self.forward(batch_seqs, batch_seqs_time, batch_last_time)  #, output1, output2

        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_func(logits, batch_last_items)


        # NCE
        aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs)
        seq_output1 = self.forward(aug_seq1, batch_seqs_time, batch_last_time)
        seq_output2 = self.forward(aug_seq2, batch_seqs_time, batch_last_time)
        org_embed, output = self.item_time_disruption(batch_seqs,seq_output)

        cl_loss = self.lmd * self.info_nce(
            seq_output1, seq_output2, temp=self.tau, batch_size=aug_seq1.shape[0])

       # cl_loss1 = self.lmd * self.info_nce(
            # output1, output2, temp=self.tau, batch_size=batch_seqs.shape[0])
        cl_loss1 = self.lmd * self.info_nce(
            org_embed, output, temp=self.tau, batch_size=batch_seqs.shape[0])


        loss_dict = {
            'rec_loss': loss.item(),
             'rec_loss1': cl_loss1.item(),
            'cl_loss': cl_loss.item(),
        }
        #return loss  + cl_loss +0.0001* cl_loss1, loss_dict
        return loss  + cl_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, batch_last_items, batch_seqs_time, batch_last_time  = batch_data
        logits = self.forward(batch_seqs, batch_seqs_time, batch_last_time)
        test_item_emb = self.emb_layer.token_emb.weight[:self.item_num + 1]
        scores = torch.matmul(logits, test_item_emb.transpose(0, 1))
        return scores
