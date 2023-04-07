import torch
import torch.nn as nn
from torch import Tensor
from .transformer_layers import TransformerEncoderLayer
from .Embedding import Sentence_Embeddings, Text_Embedding
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size
        :return:
        """
        return self._output_size

class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size
        self._hidden_size = hidden_size

        if freeze:
            freeze_params(self)

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = embed_src

        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)

class T2BEncoder(nn.Module):
    def __init__(self, vocab_size=204, obj_classes_size=154, hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1, cfg=None):
        super(T2BEncoder, self).__init__()
        self.input_embeddings = Text_Embeddings(vocab_size, obj_classes_size, hidden_size, max_rel_pair= 33)
        self.encoder = TransformerEncoder(hidden_size=hidden_size, ff_size=hidden_size * 4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, emb_dropout=dropout)
        self.hidden_size = hidden_size
        self.vocab_classifier = nn.Linear(hidden_size, vocab_size)
        self.obj_id_classifier = nn.Linear(hidden_size, obj_classes_size)
        self.token_type_classifier = nn.Linear(hidden_size, 4)

    def forward(self, input_token, src_mask):
        batch_size = input_token.shape[0]
        
        src, class_embeds = self.input_embeddings(input_token)

        encoder_output = self.encoder(src, src_mask)

        vocab_logits = self.vocab_classifier(encoder_output)
        obj_id_logits = self.obj_id_classifier(encoder_output)
        token_type_logits = self.token_type_classifier(encoder_output)

        return encoder_output, vocab_logits, obj_id_logits, token_type_logits, src, class_embeds


class RelEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size=204, obj_classes_size=154, hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1, cfg=None):
        super(RelEncoder, self).__init__()

        self.input_embeddings = Sentence_Embeddings(vocab_size, obj_classes_size, hidden_size, max_rel_pair= 33)
        
        self.encoder = TransformerEncoder(hidden_size=hidden_size, ff_size=hidden_size * 4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, emb_dropout=dropout)
        self.hidden_size = hidden_size

        self.vocab_classifier = nn.Linear(hidden_size, vocab_size)
        self.obj_id_classifier = nn.Linear(hidden_size, obj_classes_size)
        self.token_type_classifier = nn.Linear(hidden_size, 4)

    def forward(self, input_token, input_obj_id, segment_label, token_type, src_mask):
        batch_size = input_token.shape[0]
        
        src, class_embeds = self.input_embeddings(input_token, 
                                        input_obj_id, segment_label, token_type)

        encoder_output = self.encoder(src, src_mask)

        vocab_logits = self.vocab_classifier(encoder_output)
        obj_id_logits = self.obj_id_classifier(encoder_output)
        token_type_logits = self.token_type_classifier(encoder_output)

        return encoder_output, vocab_logits, obj_id_logits, token_type_logits, src, class_embeds

