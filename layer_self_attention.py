import torch.nn as nn
import torch
import torch.nn.functional as F

class layer_self_attention(nn.Module):

    def __init__(self, in_channels, out_channels, dk, dq, dv, Nh):
        super(layer_self_attention, self).__init__()
        self.Cin = in_channels
        self.Cout = out_channels
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.Nh = Nh

        self.k = int(self.dk * self.Cin)
        self.q = int(self.dq * self.Cin)
        self.v = int(self.dv * self.Cin)

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.Cin, self.q, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.q,self.q)
        )

        self.kv_conv = nn.Sequential(
            nn.Conv2d(self.Cin, self.k + self.v, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(self.k + self.v, self.k + self.v)
        )

        self.attn = nn.Conv2d(self.v, self.Cout, kernel_size=1, stride=1)

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    #shape of flat_q: (N, Nh, dq//Nh, H*W)
    #shape of q:      (N, Nh, dq//Nh, H, W)
    def layer_compute_flat_qkv(self, x, x_st, dq, dk, dv, Nh):
        q = self.q_conv(x)
        N, _, Hq, Wq = q.shape

        kv = self.kv_conv(x_st)
        N, _, H, W = kv.shape
        k,v = torch.split(kv, [dk, dv], dim=1)


        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dq // Nh, Hq * Wq))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))

        return flat_q, flat_k, flat_v, q, k, v

#use inputs_st(k,v) to strength inputs(q)
    def forward(self, inputs, inputs_st):
        batch, N, H, W = inputs.shape
        #print(inputs.shape)
        flat_q, flat_k, flat_v, q, k, v = self.layer_compute_flat_qkv(inputs, inputs_st,self.q, self.k,self.v,self.Nh)
        #print(flat_q.shape)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=1)
        #print(weights.shape)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.v // self.Nh, H, W))
        #print(attn_out.shape)
        attn_out = torch.reshape(attn_out, (batch, self.Nh * (self.v // self.Nh), H, W))
        #print(attn_out.shape)
        attn_out = self.attn(attn_out)
        #print(attn_out.shape)

        return attn_out


