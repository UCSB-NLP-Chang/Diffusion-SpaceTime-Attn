# iccv version
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np
import pickle as pkl

from ldm.modules.diffusionmodules.util import checkpoint
from process_id import NON_EXISTING_NAME_ID

mode = "fix_radius_0p2"

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, self_attention_region=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if self_attention_region is not None:
            dim = int(np.sqrt(x.shape[1]))
            channel = x.shape[2]
            
            for each_region in self_attention_region:
                x1 = x.reshape(6,dim,dim,channel)[:,int(each_region[0]*dim):int(each_region[1]*dim), int(each_region[2]*dim):int(each_region[3]*dim), :].reshape(6, -1, channel)
                q = self.to_q(x1)
                k = self.to_k(x1)
                v = self.to_v(x1)
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
                attn = sim.softmax(dim=-1)
                out1 = einsum('b i j, b j d -> b i d', attn, v)
                out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=h)
                out.reshape(2,dim,dim,channel)[:,int(each_region[0]*dim):int(each_region[1]*dim), int(each_region[2]*dim):int(each_region[3]*dim), :] = out1.reshape(2, int(each_region[1]*dim)-int(each_region[0]*dim), int(each_region[3]*dim)-int(each_region[2]*dim), channel)

        return self.to_out(out)

import torchvision
def plot(mask_tensor, i):
    mask_tensor = mask_tensor.unsqueeze(0).repeat(3,1,1).to(torch.uint8) * 255
    torchvision.io.write_png(mask_tensor, "test%d.png"%(i))
    return

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.uncond = torch.load("uncond_%s_g0.pt"%(mode))


    def forward(self, x, context=None, time=None, text_index=None, coef=None, bboxs_curr=None):
        self.bboxs_curr = bboxs_curr
        num_objects = len(bboxs_curr)
        if time == 981: # 981 is the first timestamp for 50-step infernece. Need to change this if # step changes.
            self.curr_cs = []
            self.masks = []
            dim = int(np.sqrt(x.shape[1]))
            channel = x.shape[2]
            for i in range(num_objects):
                curr_c = torch.load("c%d_%s_g%d.pt"%(i, mode, NON_EXISTING_NAME_ID))
                curr_c = torch.cat((self.uncond, curr_c))
                self.curr_cs.append(curr_c)

                # for each dimension, prepare corresponding mask
                obj_x = bboxs_curr[i][0]
                obj_y = bboxs_curr[i][1]

                axis1 = torch.arange(dim, dtype=torch.float32) / dim
                axis2 = torch.arange(dim, dtype=torch.float32) / dim

                dist1 = (axis1 - obj_x) ** 2
                dist2 = (axis2 - obj_y) ** 2

                dist = dist1.unsqueeze(0) + dist2.unsqueeze(1)
                mask = dist < 0.04 # r = 0.2, 0.2**2 = 0.04
                mask = mask.reshape(1, dim, dim, 1).repeat(1, 1, 1, channel).to(x.device)
                self.masks.append(mask)

        
        return checkpoint(self._forward, (x, context, coef), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, coef=None, bboxs_curr_input=None):
        bboxs = self.bboxs_curr
        num_objects = len(bboxs)
        curr_cs = self.curr_cs
        gs = []
        
        x = self.attn1(self.norm1(x)) + x
        x1 = x.clone()
        dim = int(np.sqrt(x.shape[1]))
        channel = x.shape[2]
        for i in range(num_objects):
            gs.append(self.attn2(self.norm2(x), context=curr_cs[i]))

        g =  self.attn2(self.norm2(x), context=context)
        x.reshape(2,dim,dim,channel)[:,:,:,:] = g.reshape(2,dim,dim,channel)
        
        for i in range(num_objects):
            if mode == 'fix_radius_0p2':
                centroid_dim1 = bboxs[i][1]
                centroid_dim2 = bboxs[i][0]

                coefficient = coef[i]
                diff_tensor = (coefficient * gs[i]).reshape(2,dim,dim,channel)[1:] - (coefficient * g).reshape(2,dim,dim,channel)[0:1]
                # ==================
                mask_tensor = self.masks[i]
                add_tensor = mask_tensor * diff_tensor
                x.reshape(2,dim,dim,channel)[1:,:,:,:] = x.reshape(2,dim,dim,channel)[1:,:,:,:] + add_tensor
                # ==================

        x = x + x1

        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, time=None, text_index=None, coef=None, bboxs_curr=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, time=time, text_index=text_index, coef=coef, bboxs_curr=bboxs_curr)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in