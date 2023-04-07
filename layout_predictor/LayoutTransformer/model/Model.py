from .Encoder import TransformerEncoder, RelEncoder, T2BEncoder
from .Decoder import TransformerDecoder, BboxDecoder, BboxRegDecoder
from .Embedding import Sentence_Embeddings, Concat_Embeddings, Add_Embeddings
from .bbox_head import BBox_Head
from .Inference import greedy, beam_search
from .Inference_Reg import greedy_Reg
import torch.nn as nn
from torch import Tensor
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import os

from typing import Dict, List, Optional
import pickle as pkl
import torch.nn.functional as F
import math
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.models.transformer import TransformerConfig
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    transformer_layer,
)
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel
)

def module_name_fordropout(module_name: str) -> str:
    if module_name == "TransformerEncoderBase":
        return "TransformerEncoder"
    else:
        return module_name

class TransformerEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop
        self.return_fc = return_fc

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.object_embedding = nn.Parameter(torch.zeros(1,768))
        self.object_embedding = nn.init.kaiming_normal_(self.object_embedding)

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(
            cfg, return_fc=self.return_fc
        )
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        object_pos = None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, object_pos
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        object_pos = None
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        if object_pos is not None:
            # object_pos will be a B x T BOOLEAN tensor, with TRUE denotes
            #  this index has an object and FALSE denotes this index does not.
            
            object_pos = object_pos.unsqueeze(-1).repeat(1,1,768)
            x += self.object_embedding * object_pos

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # nested tensor and BT enable
        layer = self.layers[0]
        BT_flag = False
        NT_flag = False
        # torch version check, BT>=1.12.0 and NT>=1.13.0.dev20220613
        # internal format is '1.13.0a0+fb'
        # external format is '1.13.0.dev20220613'(cpu&gpu) for nightly or "1.11.0"(cpu) or '1.11.0+cu102'(gpu) for stable
        BT_version = False
        NT_version = False
        if "fb" in torch.__version__:
            BT_version = True
            NT_version = True
        else:
            if "+" in torch.__version__:
                torch_version = torch.__version__.split("+")[0]
            else:
                torch_version = torch.__version__

            torch_version = torch_version.split(".")
            int_version = (
                int(torch_version[0]) * 1000
                + int(torch_version[1]) * 10
                + int(torch_version[2])
            )
            if len(torch_version) == 3:
                if int_version >= 1120:
                    BT_version = True
                if int_version >= 1131:
                    NT_version = True
            elif len(torch_version) == 4:
                if int_version >= 1130:
                    BT_version = True
                # Consider _nested_tensor_from_mask_left_aligned is landed after "20220613"
                if int_version >= 1131 or (
                    int_version == 1130 and torch_version[3][3:] >= "20220613"
                ):
                    NT_version = True

        if (
            BT_version
            and x.dim() == 3
            and layer.load_to_BT
            and not layer.return_fc
            and layer.can_use_fastpath
            and not layer.training
            and not layer.ever_training
            and not layer.cfg_checkpoint_activations
        ):
            # Batch first can not be justified but needs user to make sure
            x = x.transpose(0, 1)
            # Check mask conditions for nested tensor
            if NT_version:
                if (
                    encoder_padding_mask is not None
                    and torch._nested_tensor_from_mask_left_aligned(
                        x, encoder_padding_mask.logical_not()
                    )
                ):
                    if not torch.is_grad_enabled() or not x.requires_grad:
                        x = torch._nested_tensor_from_mask(
                            x, encoder_padding_mask.logical_not()
                        )
                        NT_flag = True
            BT_flag = True

        # encoder layers
        if NT_flag:
            processing_mask = None
        else:
            processing_mask = encoder_padding_mask
        encoder_padding_mask_out = processing_mask if has_pads else None
        for layer in self.layers:
            lr = layer(x, encoder_padding_mask=encoder_padding_mask_out)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        # change back to non-nested and Batch second
        if NT_flag:
            x = x.to_padded_tensor(0.0)

        if NT_flag or BT_flag:
            x = x.transpose(0, 1)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    @torch.jit.export
    def _reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """Dummy re-order function for beamable enc-dec attention"""
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict



class TransformerEncoderRoberta(TransformerEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )

class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

def base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )

class RobertaEncoder(FairseqEncoder):
    """RoBERTa encoder Modified."""

    def __init__(self):
        curr_path = os.path.abspath(os.path.join(os.path.join(os.path.abspath(__file__), os.pardir), os.pardir))
        with open(curr_path + "/configs/test-args.pkl", "rb") as f:
            args = pkl.load(f)

        with open(curr_path + "/configs/test-dictionary.pkl", "rb") as f:
            dictionary = pkl.load(f)

        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens = self.build_embedding(
            len(dictionary), args.encoder_embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, embed_tokens)

        self.lm_head = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = TransformerEncoderRoberta(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        object_pos=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens, object_pos=object_pos
        )
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, object_pos=None, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
            object_pos=object_pos
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

class Layout_Transformer(nn.Module):
    """
    Base Model class
    """

    def __init__(self, input_embeddings, output_embeddings) -> None:
        super(Layout_Transformer, self).__init__()

        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.bos_index = 1
        self.pad_index = 0
        self.eos_index = 2

    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                trg_mask: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(src=src, src_mask=src_mask)
        # print("Encoder output:", encoder_output.size())
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, src: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        :param src:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.
        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=trg_input,
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    # def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
    #         -> Tensor:
    #     """
    #     Compute non-normalized loss and number of tokens for a batch
    #     :param batch: batch to compute loss for
    #     :param loss_function: loss function, computes for input and target
    #         a scalar loss for the complete batch
    #     :return: batch_loss: sum of losses over non-pad elements in the batch
    #     """
    #     # pylint: disable=unused-variable
    #     out, hidden, att_probs, _ = self.forward(
    #         src=batch.src, trg_input=batch.trg_input,
    #         src_mask=batch.src_mask, src_lengths=batch.src_lengths,
    #         trg_mask=batch.trg_mask)

    #     # compute log probs
    #     log_probs = F.log_softmax(out, dim=-1)

    #     # compute batch loss
    #     batch_loss = loss_function(log_probs, batch.trg)
    #     # return batch loss = sum over all elements in batch that are not pad
    #     return batch_loss

    def inference(self, src: Tensor, src_mask: Tensor, trg_embed: Add_Embeddings, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch
        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoder_output, encoder_hidden = self.encode(src, src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = 64

        # greedy decoding
        if beam_size < 2:
            output_cats, output_pos, output_shape = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output, eos_index=self.eos_index,
                    src_mask=src_mask, embed=trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return output_cats, output_pos, output_shape

    # def __repr__(self) -> str:
    #     """
    #     String representation: a description of encoder, decoder and embeddings
    #     :return: string representation
    #     """
    #     return "%s(\n" \
    #            "\tencoder=%s,\n" \
    #            "\tdecoder=%s,\n" \
    #            "\tsrc_embed=%s,\n" \
    #            "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
    #                self.decoder, self.src_embed, self.trg_embed)


# def build_model(cfg: dict = None,
#                 src_vocab: Vocabulary = None,
#                 trg_vocab: Vocabulary = None) -> Model:
#     """
#     Build and initialize the model according to the configuration.
#     :param cfg: dictionary configuration containing model specifications
#     :param src_vocab: source vocabulary
#     :param trg_vocab: target vocabulary
#     :return: built and initialized model
#     """
#     src_padding_idx = 0
#     trg_padding_idx = 0

#     src_embed = Embeddings(
#         **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
#         padding_idx=src_padding_idx)

#     # this ties source and target embeddings
#     # for softmax layer tying, see further below
#     if cfg.get("tied_embeddings", False):
#         if src_vocab.itos == trg_vocab.itos:
#             # share embeddings for src and trg
#             trg_embed = src_embed
#         else:
#             raise ConfigurationError(
#                 "Embedding cannot be tied since vocabularies differ.")
#     else:
#         trg_embed = Embeddings(
#             **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
#             padding_idx=trg_padding_idx)

#     # build encoder
#     enc_dropout = cfg["encoder"].get("dropout", 0.)
#     enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
#     if cfg["encoder"].get("type", "recurrent") == "transformer":
#         assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
#                cfg["encoder"]["hidden_size"], \
#                "for transformer, emb_size must be hidden_size"

class Text2Layout(nn.Module):
    """
    Base Model class
    """

    def __init__(self, input_embeddings, output_embeddings) -> None:
        super(Text2Layout, self).__init__()

        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = TransformerDecoder()
        self.bos_index = 1
        self.pad_index = 0
        self.eos_index = 2

    # pylint: disable=arguments-differ
    def forward(self, caption, trg_input, trg_mask):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.
        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoded = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=caption, max_length = 64, pad_to_max_length = True)
        input_ids = torch.tensor(encoded['input_ids']).cuda()
        attention_mask = torch.tensor(encoded['attention_mask']).bool().cuda()
        token_type_ids = torch.tensor(encoded['token_type_ids']).cuda()
        encoder_output = self.encode(input_ids, attention_mask, token_type_ids)[0]
        encoder_hidden = None
        attention_mask = attention_mask.unsqueeze(1)
        # print("Encoder output:", encoder_output.size())
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=attention_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        Encodes the source sentence.
        :param src:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.
        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=trg_input,
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)

    def inference(self, caption, max_output_length, beam_size, beam_alpha):
        """
        Get outputs and attentions scores for a given batch
        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoded = self.tokenizer.batch_encode_plus(caption, max_length = 64, pad_to_max_length = True)
        input_ids = torch.tensor(encoded['input_ids']).cuda()
        attention_mask = torch.tensor(encoded['attention_mask']).bool().cuda()
        token_type_ids = torch.tensor(encoded['token_type_ids']).cuda()
        encoder_output = self.encode(input_ids, attention_mask, token_type_ids)[0]
        encoder_hidden = None
        attention_mask = attention_mask.unsqueeze(1)
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = 64

        # greedy decoding
        if beam_size < 2:
            output_cats, output_pos, output_shape, pred_cats, pred_pos, pred_shape = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output, eos_index=self.eos_index,
                    src_mask=attention_mask, embed=self.output_embeddings,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=attention_mask, embed=self.output_embeddings,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder)

        return output_cats, output_pos, output_shape, pred_cats, pred_pos, pred_shape

class Rel2Layout(nn.Module):

    def __init__(self, vocab_size=204, cls_size=154, pos_size=68, shape_size=68,\
                 hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1):
        super(Rel2Layout, self).__init__()

        self.encoder = RelEncoder(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, \
            attn_heads=attn_heads, dropout=dropout)
        self.decoder = BboxDecoder(cls_size=cls_size, pos_size=pos_size, shape_size=shape_size,\
            hidden_size=hidden_size, num_layers=num_layers, attn_heads=attn_heads, dropout=dropout)

        self.hidden_size = hidden_size
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2        

    def forward(self, input_token, input_ids, segment_label, token_type, src_mask, output_cls, output_pos, output_shape, trg_mask):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        return self.decoder(output_cls, output_pos, output_shape, encoder_output, src_mask, trg_mask)

    def inference(self, input_token, input_ids, segment_label, token_type, src_mask):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        max_output_length = 64

        # greedy decoding
        return greedy(encoder_hidden=None, encoder_output=encoder_output, eos_index=self.eos_index, \
            src_mask=src_mask, bos_index=self.bos_index, \
            decoder=self.decoder, max_output_length=max_output_length)

class Rel2RegLayout(nn.Module):

    def __init__(self, vocab_size=204, cls_size=154, box_size=4,
                 hidden_size=512, num_layers=6, max_out_len = 128, attn_heads=8, dropout=0.1):
        super(Rel2RegLayout, self).__init__()

        self.encoder = RelEncoder(vocab_size=vocab_size, hidden_size=hidden_size,
                                  num_layers=num_layers,attn_heads=attn_heads,
                                  dropout=dropout)
        self.decoder = BboxRegDecoder(cls_size=cls_size, box_size=box_size,
                                      hidden_size=hidden_size, num_layers=num_layers,
                                      attn_heads=attn_heads, dropout=dropout)

        self.hidden_size = hidden_size
        self.max_out_len = max_out_len
        self.pad_index = 0
        self.bos_index = 1
        self.eos_index = 2        

    def forward(self, input_token, input_ids, segment_label, token_type, src_mask,
                output_cls, output_box, trg_mask, trg_input_template):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)
        return self.decoder(output_cls, output_box, trg_input_template,
                            encoder_output, src_mask, trg_mask)

    def inference(self, input_token, input_ids, segment_label, token_type, src_mask,
                  trg_input_template):

        src = self.encoder.input_embeddings(input_token, input_ids, segment_label, token_type)

        encoder_output = self.encoder.encoder(src, src_mask)

        max_output_length = self.max_out_len
        # greedy decoding
        return greedy_Reg(encoder_hidden=None, encoder_output=encoder_output,
                          eos_index=self.eos_index, src_mask=src_mask, 
                          bos_index=self.bos_index, decoder=self.decoder, 
                          max_output_length=max_output_length, 
                          template = trg_input_template)

class Rel2Bbox(nn.Module):
    def __init__(self, vocab_size=204, obj_classes_size=154, noise_size=64,\
                 hidden_size=256, num_layers=4, attn_heads=4, dropout=0.1, cfg=None):
        super(Rel2Bbox, self).__init__()
        self.encoder = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.encoder.model.encoder = RobertaEncoder()
        self.bbox_head = BBox_Head(hidden_size=hidden_size, dropout=dropout, cfg=cfg)

    def forward(self, bpe_toks_tensor, src_masks_tensor, bpe_label_pair,
                inference=False, epoch=0, trg_mask=None, global_mask=None, object_pos_tensor=None):
        # x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, object_pos=object_pos,**kwargs)
        features, _ = self.encoder.model.encoder(bpe_toks_tensor, object_pos=object_pos_tensor.to(bpe_toks_tensor.device))    # B x L x Hidden_dim (1024) 
        if inference:
            coarse_xy, coarse_gmm_xy, _, _ = self.bbox_head.inference(features, src_masks_tensor, trg_mask, global_mask)
        else:
            coarse_xy, coarse_gmm_xy, _, _ = self.bbox_head(epoch, features, src_masks_tensor, trg_mask, global_mask)

        return coarse_xy, coarse_gmm_xy, None, None
