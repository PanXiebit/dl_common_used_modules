# -*- encoding = "utf-8" -*-

import torch
import torch.nn as nn
class TriAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TriAttention, self).__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        """
        :param C: [batch, c_len, d_model]
        :param Q: [batch, q_len, d_model]
        :param Cmask:  [batch, c_len]
        :param Qmask:  [batch, q_len]
        :return:
        """
        # C = C.transpose(1, 2)
        # Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)  # [batch, c_len, q_len]
        Cmask = Cmask.view(batch_size_c, Lc, 1)  # [batch, c_len, 1]
        Qmask = Qmask.view(batch_size_c, 1, Lq)  # [batch, 1, q_len]
        S_mask = mask_logits(mask_logits(S, Qmask), Cmask)
        S1 = F.softmax(S_mask, dim=2)
        S2 = F.softmax(S_mask, dim=1)
        # S1 = F.softmax(mask_logits(S_mask, Qmask), dim=2)
        # S2 = F.softmax(mask_logits(S_mask, Cmask), dim=1)
        A = torch.bmm(S1, Q)  # [batch, c_len, d_model]
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)  # [batch, c_len, d_model]
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        # return out.transpose(1, 2)
        out = mask_logits(out, Cmask)
        return out

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])  # [batch, c_len, q_len]
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])  # [batch, c_len, q_len]
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))  # [batch, c_len, q_len]
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

