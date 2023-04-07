import torch
import torch.nn as nn
from torch import Tensor
from .transformer_layers import TransformerDecoderLayer, CustomTransformerDecoderLayer
import numpy as np
from .Embedding import Concat_Embeddings, Add_Embeddings, ConcatBox_Embeddings

class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)
        :return:
        """
        return self._output_size

class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 src_trg_att: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout,src_trg_att=src_trg_att) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = trg_embed

        trg_mask = trg_mask & self.subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output,
                      src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        return x

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)

    def subsequent_mask(self, size: int) -> Tensor:
        """
        Mask out subsequent positions (to prevent attending to future positions)
        Transformer helper function.
        :param size: size of mask (2nd and 3rd dim)
        :return: Tensor with 0s and 1s of shape (1, size, size)
        """
        mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(mask) == 0

class CustomTransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_bb_size: int = 64,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(CustomTransformerDecoder, self).__init__()

        self._hidden_size = hidden_size
        
        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(
                size=hidden_size, bb_size=hidden_bb_size, ff_size=ff_size, 
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size*2, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed_0: Tensor = None,
                trg_embed_1: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x_0 = trg_embed_0 # spatial
        x_1 = trg_embed_1 # semantic

        trg_mask = trg_mask & self.subsequent_mask(
            trg_embed_0.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(spatial_x=x_0, semantic_x=x_1, memory=encoder_output,
                      src_mask=src_mask, trg_mask=trg_mask)
        x = self.layer_norm(x)
        return x

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)

    def subsequent_mask(self, size: int) -> Tensor:
        """
        Mask out subsequent positions (to prevent attending to future positions)
        Transformer helper function.
        :param size: size of mask (2nd and 3rd dim)
        :return: Tensor with 0s and 1s of shape (1, size, size)
        """
        mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(mask) == 0
class BboxDecoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, cls_size=154, pos_size=68, shape_size=68,\
                 hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1):
        super(BboxDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_embeddings = Add_Embeddings(cls_size, pos_size, shape_size, hidden_size, dropout)
        self.decoder = TransformerDecoder(hidden_size=hidden_size, ff_size=hidden_size*4, num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, emb_dropout=dropout, vocab_size=cls_size, max_pos=pos_size, \
            max_shape_type=shape_size)
        self.latent_transformer = nn.Linear(hidden_size, hidden_size)
        self.cls_classifier = nn.Linear(hidden_size, cls_size)
        self.pos_classifier = nn.Linear(hidden_size, pos_size)
        self.shape_classifier = nn.Linear(hidden_size, shape_size)
        
        nn.init.normal_(self.cls_classifier.weight, std=0.01)
        nn.init.normal_(self.pos_classifier.weight, std=0.01)
        nn.init.normal_(self.shape_classifier.weight, std=0.01)
        for l in [self.cls_classifier, self.pos_classifier, self.shape_classifier]:
            nn.init.constant_(l.bias, 0)

    def forward(self, output_cls, output_pos, output_shape, encoder_output, src_mask, trg_mask):

        encoder_output = self.latent_transformer(encoder_output)
        trg_input = self.output_embeddings(output_cls, output_pos, output_shape)
        unroll_steps = trg_input.size(1)

        decoder_output = self.decoder(trg_embed=trg_input, encoder_output=encoder_output, encoder_hidden=None,\
                    src_mask=src_mask, unroll_steps=unroll_steps, hidden=None, trg_mask=trg_mask)

        output_cats = self.cls_classifier(decoder_output)
        output_pos = self.pos_classifier(decoder_output)
        output_shape = self.shape_classifier(decoder_output)

        return output_cats, output_pos, output_shape


class BboxRegDecoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, cls_size=154, box_size=4,
                 hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1):
        
        super(BboxRegDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_embeddings = ConcatBox_Embeddings(cls_size, box_size, 
                                                      hidden_size, dropout)
        self.decoder = TransformerDecoder(hidden_size=hidden_size, ff_size=hidden_size*4,
                                          num_layers=num_layers,num_heads=attn_heads, 
                                          dropout=dropout, emb_dropout=dropout,
                                          vocab_size=cls_size, box_size=box_size)
        self.cls_classifier = nn.Linear(hidden_size, cls_size)
        self.box_predictor = nn.Sequential(
            nn.Linear(hidden_size, 4),
            nn.Sigmoid()
        )        
    def forward(self, output_cls, output_box, output_template,
                encoder_output, src_mask, trg_mask):

        trg_input = self.output_embeddings(output_cls, output_box, output_template)
        unroll_steps = trg_input.size(1)

        decoder_output = self.decoder(trg_embed=trg_input, encoder_output=encoder_output,
                                      encoder_hidden=None,src_mask=src_mask,
                                      unroll_steps=unroll_steps, hidden=None, 
                                      trg_mask=trg_mask)

        output_cats = self.cls_classifier(decoder_output)
        output_boxes = self.box_predictor(decoder_output)

        return output_cats, output_boxes
    
class PDFDecoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, cls_size=154, box_size=4,
                 hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1):
        
        super(BboxRegDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.decoder = TransformerDecoder(hidden_size=hidden_size, ff_size=hidden_size*4,
                                          num_layers=num_layers,num_heads=attn_heads, 
                                          dropout=dropout, emb_dropout=dropout,
                                          vocab_size=cls_size, box_size=box_size)

    def forward(self, output_box_embed, encoder_output, src_mask, trg_mask):

        trg_input = output_box_embed
        unroll_steps = trg_input.size(1)

        decoder_output = self.decoder(trg_embed=trg_input, encoder_output=encoder_output,
                                      encoder_hidden=None,src_mask=src_mask,
                                      unroll_steps=unroll_steps, hidden=None, 
                                      trg_mask=trg_mask)
        return decoder_output



