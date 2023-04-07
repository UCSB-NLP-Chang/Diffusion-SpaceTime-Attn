import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


# pylint: disable=unused-argument
def greedy_PDF(
        src_mask, bos_index, eos_index,
        max_output_length, decoder,
        encoder_output, encoder_hidden, global_mask):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.
    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding layer
    :param bos_index: index of <s> in the vocabulary
    :param eos_index: index of </s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    gmm_comp_num = 5
    batch_size = src_mask.size(0)
    ys_1 = encoder_output
    trg_mask = src_mask.new_ones([1, 1, 1])

    with torch.no_grad():
        _, _, sample_xy, _, xy_gmm, _ = decoder(
                                encoder_output[:,:-1,:], ys_1, src_mask, trg_mask, is_train=False, global_mask=global_mask)

    return None, None, sample_xy, None, xy_gmm, None

def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
