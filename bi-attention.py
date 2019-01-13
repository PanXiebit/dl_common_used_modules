

import torch
import torch.nn as nn
import torch.nn.functional as F
# from modules.layers import mask_logits

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?


class BiAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1, multi_head=False, heads=4):
        super(BiAttention, self).__init__()
        self.d_model = d_model
        self.multi_head = multi_head
        self.heads = heads
        if not multi_head:
            self.W1 = nn.Linear(d_model, d_model)
            self.Wf = nn.Linear(4 * d_model, 4 * d_model, bias=True)
            self.Wg = nn.Linear(4 * d_model, 4 * d_model, bias=True)
        else:
            d_model = int(d_model/heads)
            self.W1 = nn.Linear(d_model, d_model)
            self.Wf = nn.Linear(4 * d_model, 4 * d_model, bias=True)
            self.Wg = nn.Linear(4 * d_model, 4 * d_model, bias=True)
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.Wf.weight)
        nn.init.xavier_normal_(self.Wg.weight)

    def forward(self, C, Q, Cmask, Qmask):
        """

        :param C:  [batch, c_len, d_model] or [batch, heads, c_len, d_model]
        :param Q:  [batch, q_len, d_model] or [batch, heads, c_len, d_model]
        :param maskC: [batch, c_len]
        :param maskQ: [batch, q_len]
        :return:
        """
        if self.multi_head:
            if (self.d_model % self.heads != 0):
                raise IndexError("d_model must be an integer multiple of heads")
            C = self.split_last_dim(C, self.heads)
            Q = self.split_last_dim(Q, self.heads)

        if len(C.size()) == 3 and len(Q.size()) == 3:
            # batch_size_c = C.size()[0]
            batch_size, Lc, d_model = C.shape
            batch_size, Lq, d_model = Q.shape
            S = self.bilinear_for_attention(C, Q)      # [batch, c_len, q_len]
            Cmask = Cmask.view(batch_size, Lc, 1)    # [batch, c_len, 1]
            Qmask = Qmask.view(batch_size, 1, Lq)
            mask_S = mask_logits(S, Cmask)
            mask_S = mask_logits(mask_S, Qmask)

            S_ = F.softmax(mask_S, dim=2)  # [batch, c_len, q_len]
            attn_C = S_.bmm(Q)             # [batch, c_len, d_model]
            attn_C = torch.cat([C, attn_C, attn_C-C, attn_C * C], dim=2)   # [batch, c_len, d_model*4]
            fuse_C = torch.tanh(self.Wf(attn_C))
            gate_C = torch.sigmoid(self.Wg(attn_C))
            out = gate_C * fuse_C + (1-gate_C) * attn_C
            out = mask_logits(out, Cmask)

        elif len(C.size()) == 4 and len(Q.size()) == 4:
            batch_size, heads, Lc, d_model = C.shape
            batch_size, heads, Lq, d_model = Q.shape
            S = self.bilinear_for_attention(C, Q)     # [batch, heads, c_len, q_len]
            Cmask = Cmask.view(batch_size, 1, Lc, 1)  # [batch, 1, c_len, 1]
            Qmask = Qmask.view(batch_size, 1, 1, Lq)
            mask_S = mask_logits(S, Cmask)
            mask_S = mask_logits(mask_S, Qmask)
            S_ = F.softmax(mask_S, dim=-1)
            attn_C = S_.matmul(Q)            # [batch, heads, c_len, d_model]
            attn_C = torch.cat([C, attn_C, attn_C - C, attn_C * C], dim=-1)  # [batch, heads, c_len, d_model*4]
            fuse_C = torch.tanh(self.Wf(attn_C))
            gate_C = torch.sigmoid(self.Wg(attn_C))
            out = gate_C * fuse_C + (1 - gate_C) * attn_C   # [batch, heads, c_len, d_model]
            out = mask_logits(out, Cmask)
        else:
            raise IndexError("the size of query and passage must be "
                             "[batch, len, d_model] or [batch, heads, len, d_model]")

        if self.multi_head:
            out = out.permute(0, 2, 1, 3)
            out = self.combine_last_two_dim(out)
        return out

    def bilinear_for_attention(self, C, Q):
        C_ = F.leaky_relu(self.W1(C))  # [batch, c_len, d_model]
        Q_ = F.leaky_relu(self.W1(Q))  # [batch, q_len, d_model]
        if len(C.size()) == 3 and len(Q.size()) == 3:
            sim_mat = C_.bmm(Q_.transpose(2,1))   # [batch, c_len, q_len]
        else:
            sim_mat = torch.matmul(C_, Q_.permute(0, 1, 3, 2))
        return sim_mat

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)  # [batch, heads, c_len, d_model/heads]

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

if __name__ == "__main__":
    # test mask
    attn = CQAttention(16, multi_head=True, heads=4)
    C = torch.rand(2, 5, 16)
    C[0, 3:, :] = 0
    C[1, 2:, :] = 0
    Q = torch.rand(2, 8, 16)
    Cmask = torch.Tensor([[1,1,1,0,0],[1,1,0,0,0]])
    Qmask = torch.Tensor([[1,1,1,1,1,0,0,0], [1,1,1,1,1,1,1,0]])
    out = attn(C, Q, Cmask, Qmask)
    print(out.shape)
