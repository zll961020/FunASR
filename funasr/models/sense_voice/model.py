import tempfile
import requests
import codecs
import os
import re
from functools import lru_cache
from typing import Iterable, Optional
import types
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from funasr.metrics.compute_acc import compute_accuracy, th_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.ctc.ctc import CTC

from funasr.register import tables


from funasr.models.paraformer.search import Hypothesis
from .utils.ctc_alignment import ctc_forced_align
import logging 
from funasr.utils.context_graph import ContextGraph 
from funasr.models.sanm.attention import MultiHeadedAttentionCrossAtt
from funasr.models.contextual_paraformer.decoder import ContextualBiasDecoder
from funasr.models.scama import utils as myutils
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from typing import List
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from funasr.models.sense_voice.context_module import ContextModule

class SinusoidalPositionEncoder(torch.nn.Module):
    """ """

    def __int__(self, d_model=80, dropout_rate=0.1):
        pass

    def encode(
        self, positions: torch.Tensor = None, depth: int = None, dtype: torch.dtype = torch.float32
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (
            depth / 2 - 1
        )
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(
            inv_timescales, [1, 1, -1]
        )
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)

        return x + position_encoding


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
        lora_list=None,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        # self.linear_q = nn.Linear(n_feat, n_feat)
        # self.linear_k = nn.Linear(n_feat, n_feat)
        # self.linear_v = nn.Linear(n_feat, n_feat)

        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time1, d_k)
        k_h = torch.reshape(k, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)
        v_h = torch.reshape(v, (b, t, self.h, self.d_k)).transpose(
            1, 2
        )  # (batch, head, time2, d_k)

        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask, mask_att_chunk_encoder=None):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder

            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)

            min_value = -float(
                "inf"
            )  # float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, x, mask, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)

                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache_tmp = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
                cache = cache_tmp
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerSANM, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(self, x, cache=None, chunk_size=None, look_back=0):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, cache


@tables.register("encoder_classes", "SenseVoiceEncoderSmall")
class SenseVoiceEncoderSmall(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens


@tables.register("model_classes", "SenseVoiceSmall")
class SenseVoiceSmall(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), input_size
        )
        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        self.beam_search = None 
        # add ctc_weight
        self.ctc_weight = 1.0 
        inner_dim = kwargs.get("inner_dim", encoder_output_size)
        bias_encoder_type = kwargs.get("bias_encoder_type", "lstm")
       
        bias_encoder_dropout_rate = kwargs.get("bias_encoder_dropout_rate", 0.0)

        if bias_encoder_type == "lstm":
            self.bias_encoder = torch.nn.LSTM(
                inner_dim, inner_dim, 1, batch_first=True, dropout=bias_encoder_dropout_rate
            )
            self.bias_embed = torch.nn.Embedding(self.vocab_size, inner_dim)
        elif bias_encoder_type == "mean":
            self.bias_embed = torch.nn.Embedding(self.vocab_size, inner_dim)
        else:
            logging.error("Unsupport bias encoder type: {}".format(bias_encoder_type))
        attention_dim = encoder_output_size
        attention_heads = kwargs.get("attention_heads", 4)
        src_attention_dropout_rate = kwargs.get("src_attention_dropout_rate", 0.0)
        dropout_rate = kwargs.get("dropout_rate", 0.1)
        self.bias_decoder = ContextualBiasDecoder(
            size=attention_dim,
            src_attn=MultiHeadedAttentionCrossAtt(
                attention_heads, attention_dim, src_attention_dropout_rate
            ),
            dropout_rate=dropout_rate,
            normalize_before=True,
        )
        self.bias_output = torch.nn.Conv1d(attention_dim * 2, attention_dim, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.hotword_norm = LayerNorm(attention_dim)
        self.clas_scale = kwargs.get("clas_scale", 1.0)

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        hotword_pad = kwargs.get("hotword_pad", None)
        hotword_lengths = kwargs.get("hotword_lengths", None)
        logging.info(f'hotword_pad: {hotword_pad.shape} hotword_lengths: {hotword_lengths.shape}')
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()
        enc_content = encoder_out[:, 4:, :]
        if hotword_pad is not None and hotword_lengths is not None:
           
            # (1) 计算热词嵌入序列
            hw_seq = self.bias_embed(hotword_pad)                       # (B,H,D)
            hw_seq, (_, _) = self.bias_encoder(hw_seq)                      # (B,H,D)
            # ② 取最后一个 token 的隐藏向量当整词表征
            last_idx = (hotword_lengths.to(hw_seq.device) - 1).clamp(min=0)
            idx = torch.arange(hw_seq.size(0), device=hw_seq.device)
            hw_emb = hw_seq[idx, last_idx, :]                           # (N, D)
            # ③ 复制到 batch 维：(B, N, D)
            hw_emb = hw_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            # ---------- 2. 构造 mask ----------
            memory_mask = torch.ones(
                batch_size, 1, hw_emb.size(1), dtype=torch.bool, device=enc_content.device
            )
            tgt_mask = torch.ones(                                   # no padding in enc_content
                enc_content.size(0), enc_content.size(1), 1,
                 dtype=torch.bool, device=enc_content.device
             )

            attn_out, *_ = self.bias_decoder(
                enc_content, tgt_mask, hw_emb, memory_mask=memory_mask
            )                                                    
           
            merged = torch.cat([enc_content, attn_out * self.clas_scale], dim=2)  # (B,T,2D)
            merged = self.bias_output(merged.transpose(1, 2)).transpose(1, 2)                   # (B,T,D)
            enc_content = self.hotword_norm(enc_content + self.dropout(merged))
        # 更新回整体 encoder_out
        encoder_out = torch.cat([encoder_out[:, :4, :], enc_content], dim=1)
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])

        loss = loss_ctc + loss_rich
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [
                [
                    (
                        self.lid_int_dict[int(lid)]
                        if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict
                        else 0
                    )
                ]
                for lid in text[:, 0]
            ]
        ).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
        ).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
       
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 2. Compute attention loss
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich
    
    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.paraformer.search import BeamSearchPara
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos, context_graph=self.context_graph)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        #logging.info(f' token list: {token_list}')  
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )
        if self.context_graph and (self.ctc is None or kwargs.get("decoding_ctc_weight", 0.0) < 0.00001):
            logging.info(f'init contextscorer')
            from funasr.models.transformer.scorers.context_scorer import ContextScorer
            scorers.update(
                ctx=ContextScorer(self.context_graph)
            )
        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight"),
            ctc=kwargs.get("decoding_ctc_weight", 0.0),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
            ctx=1.0, # fix ctx weight, just adjust context_graph_score
        )
        beam_search = BeamSearchPara(
            beam_size=kwargs.get("beam_size", 2),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=kwargs.get("pre_beam_score_key", None if self.ctc_weight == 1.0 else "full"),
            pre_beam_ratio=kwargs.get("pre_beam_ratio", 1.5),
            tokenizer=kwargs.get("tokenizer", None),
            #pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
            #pre_beam_score_key='full',
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # Initialize beam search if needed
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc is not None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        logging.info(f"is_use_ctc: {is_use_ctc}, is_use_lm: {is_use_lm} , beam_search: {self.beam_search}") 
        logging.info(f"decoding_ctc_weight: {kwargs.get('decoding_ctc_weight')}, ctc: {self.ctc}")
        logging.info(f'context_graph_score: {kwargs.get("context_graph_score")} context_list_path: {kwargs.get("context_list_path")}')
        # init hotword_list 
        self.hotword_list = self.generate_hotwords_list(
                    kwargs.get("context_list_path", None), tokenizer=tokenizer, frontend=frontend)
        
        if self.beam_search is None and (is_use_lm or is_use_ctc):
            self.context_graph = None 
            #if kwargs.get("context_graph_score", 0.0) > 0.00001 and kwargs.get("context_list_path", None):
            if kwargs.get("context_list_path", None):
                logging.info(f'enable context_graph')
                context_list_path = kwargs.get("context_list_path")
                context_graph_score = kwargs.get("context_graph_score")
                token_list = kwargs.get("token_list", None)
                logging.info(f'token_list len: {len(token_list)}')
                symbol_table = {}
                # for i, t in enumerate(token_list):
                #     if i <= 10:
                #         logging.info(f'token_list: {t} len: {len(t)} i: {i}')
                #     symbol_table[t] = i
                # hotword
              
                logging.info(f"hotword_list: {self.hotword_list}")
                for hotword in self.hotword_list:
                    for i, t in enumerate(hotword):
                        logging.info(f'hotword_cur: {hotword} t: {t} token: {token_list[t]}')
                        symbol_table[token_list[t]] = t 
                self.context_graph = ContextGraph(self.hotword_list, symbol_table, context_graph_score, tokenizer)
            logging.info(f"enable beam_search is tokenizer SentencepiecesTokenizer: {isinstance(tokenizer, SentencepiecesTokenizer)}")
            kwargs["tokenizer"] = tokenizer
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)
        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        output_timestamp = kwargs.get("output_timestamp", False)

        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        if kwargs.get('is_train_hotword', False):
            logging.info(f'using trained hotword module')
            hw_list = self.hotword_list
            if hw_list is None:
                hw_list = [torch.Tensor([1]).long().to(encoder_out.device)]  # empty hotword list
                hw_list_pad = pad_list(hw_list, 0)
                hw_embed = self.bias_embed(hw_list_pad)
                hw_embed, (h_n, _) = self.bias_encoder(hw_embed)
                hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)
            else:
                hw_lengths = [len(i) for i in hw_list]
                hw_list_pad = pad_list([torch.Tensor(i).long() for i in hw_list], 0).to(
                    encoder_out.device
                )
               
                hw_embed = self.bias_embed(hw_list_pad)
                hw_embed = torch.nn.utils.rnn.pack_padded_sequence(
                    hw_embed, hw_lengths, batch_first=True, enforce_sorted=False
                )
                _, (h_n, _) = self.bias_encoder(hw_embed)
                hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)
            contextual_info = hw_embed 
            enc_content = encoder_out[:, 4:, :]                              # 去掉 4 个富标签 token
            attn_out, *_ = self.bias_decoder(
                enc_content,
                torch.ones(enc_content.size(0), enc_content.size(1), 1, dtype=torch.bool, device=enc_content.device),
                contextual_info,
                memory_mask=torch.ones(
                    contextual_info.size(0), 1, contextual_info.size(1), dtype=torch.bool, device=enc_content.device
                ),
            )
            merged      = torch.cat([enc_content, attn_out * self.clas_scale], dim=2)          # (B,T,2D)
            merged      = self.bias_output(merged.transpose(1, 2)).transpose(1, 2)
            enc_content = self.hotword_norm(enc_content + merged)

            # 拼回整体 encoder_out
            encoder_out = torch.cat([encoder_out[:, :4, :], enc_content], dim=1)
            logging.info(f'attn_out mean {attn_out.abs().mean()}')
            logging.info(f'bias_output weight norm {self.bias_output.weight.data.norm()}')
        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i].item(), :]
            am_scores = ctc_logits[i, : encoder_out_lens[i].item(), :]
            #x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            if self.beam_search is not None:
                if not self.beam_search.do_pre_beam:
                    am_scores = torch.zeros_like(
                        ctc_logits[i, : encoder_out_lens[i].item(), :]
                    )
                
                # Use beam search
                nbest_hyps = self.beam_search(
                    x=x,
                    am_scores=am_scores,
                    maxlenratio=kwargs.get("maxlenratio", 0.0),
                    minlenratio=kwargs.get("minlenratio", 0.0),
                )
                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                yseq = am_scores.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
                logging.info(f'greedy search yseq: {yseq} score: {score} decode text: {tokenizer.decode(yseq.tolist())}')
            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                token_int = list(   
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                text = tokenizer.decode(token_int)

                # result_i = {"key": key[i], "text": text}
                # results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["text"][key[i]] = text

                if output_timestamp:
                    from itertools import groupby

                    timestamp = []
                    tokens = tokenizer.text2tokens(text)[4:]
                    token_back_to_id = tokenizer.tokens2ids(tokens)
                    token_ids = []
                    for tok_ls in token_back_to_id:
                        if tok_ls: token_ids.extend(tok_ls)
                        else: token_ids.append(124)

                    if len(token_ids) == 0:
                        result_i = {"key": key[i], "text": text}
                        results.append(result_i)
                        continue

                    logits_speech = self.ctc.softmax(encoder_out)[i, 4 : encoder_out_lens[i].item(), :]
                    pred = logits_speech.argmax(-1).cpu()
                    logits_speech[pred == self.blank_id, self.blank_id] = 0
                    align = ctc_forced_align(
                        logits_speech.unsqueeze(0).float(),
                        torch.Tensor(token_ids).unsqueeze(0).long().to(logits_speech.device),
                        (encoder_out_lens[i] - 4).long(),
                        torch.tensor(len(token_ids)).unsqueeze(0).long().to(logits_speech.device),
                        ignore_id=self.ignore_id,
                    )
                    pred = groupby(align[0, : encoder_out_lens[i]])
                    _start = 0
                    token_id = 0
                    ts_max = encoder_out_lens[i] - 4
                    for pred_token, pred_frame in pred:
                        _end = _start + len(list(pred_frame))
                        if pred_token != 0:
                            ts_left = max((_start * 60 - 30) / 1000, 0)
                            ts_right = min((_end * 60 - 30) / 1000, (ts_max * 60 - 30) / 1000)
                            timestamp.append([tokens[token_id], ts_left, ts_right])
                            token_id += 1
                        _start = _end
                    timestamp, words = self.post(timestamp)
                    result_i = {"key": key[i], "text": text, "timestamp": timestamp, "words": words}
                    results.append(result_i)
                else:
                    result_i = {"key": key[i], "text": text}
                    results.append(result_i)
        return results, meta_data

    def post(self, timestamp):
        timestamp_new = []
        words_new = []
        prev_word = None
        for i, t in enumerate(timestamp):
            word, start, end = t
            start = int(start * 1000)
            end = int(end * 1000)
            if word == "▁":
                continue
            if i == 0:
                # timestamp_new.append([word, start, end])
                timestamp_new.append([start, end])
                words_new.append(word)
            elif word.startswith("▁"):
                word = word[1:]
                timestamp_new.append([start, end])
                words_new.append(word)
            elif prev_word is not None and prev_word.isalpha() and prev_word.isascii() and word.isalpha() and word.isascii():
                word = prev_word + word
                timestamp_new[-1][1] = end
                words_new[-1] = word
            else:
                # timestamp_new[-1][0] += word
                timestamp_new.append([start, end])
                words_new.append(word)
            prev_word = word
        return timestamp_new, words_new

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models

        return results, meta_data

    def generate_hotwords_list(self, hotword_list_or_file, tokenizer=None, frontend=None):

        def seg_tokenize(txt, seg_dict):
            pattern = re.compile(r"^[\u4E00-\u9FA50-9]+$")
            out_txt = ""
            for word in txt:
                word = word.lower()
                if word in seg_dict:
                    out_txt += seg_dict[word] + " "
                else:
                    if pattern.match(word):
                        for char in word:
                            if char in seg_dict:
                                out_txt += seg_dict[char] + " "
                            else:
                                out_txt += "<unk>" + " "
                    else:
                        out_txt += "<unk>" + " "
            return out_txt.strip().split()

        seg_dict = None
        if frontend.cmvn_file is not None:
            model_dir = os.path.dirname(frontend.cmvn_file)
            seg_dict_file = os.path.join(model_dir, "seg_dict")
            if os.path.exists(seg_dict_file):
                seg_dict = load_seg_dict(seg_dict_file)
            else:
                seg_dict = None
        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    res = tokenizer.tokens2ids(hw_list)
                    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
                        res = res[0]
                    hotword_list.append(res)
                #hotword_list.append([self.sos])
                #hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for url, download and generate txt
        elif hotword_list_or_file.startswith("http"):
            logging.info("Attempting to parse hotwords from url...")
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            text_file_path = os.path.join(work_dir, os.path.basename(hotword_list_or_file))
            local_file = requests.get(hotword_list_or_file)
            open(text_file_path, "wb").write(local_file.content)
            hotword_list_or_file = text_file_path
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    hotword_list.append(tokenizer.tokens2ids(hw_list))
                hotword_list.append([self.sos])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for text str input
        elif not hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hw_list = hw.strip().split()
                if seg_dict is not None:
                    hw_list = seg_tokenize(hw_list, seg_dict)
                hotword_list.append(tokenizer.tokens2ids(hw_list))
            hotword_list.append([self.sos])
            hotword_str_list.append("<s>")
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list
    
@tables.register("model_classes", "SenseVoiceSmallCPNN")
class SenseVoiceSmallCPNN(nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        ctc_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        if ctc_conf is None:
            ctc_conf = {}
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **ctc_conf)

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.error_calculator = None

        self.ctc = ctc

        self.length_normalized_loss = length_normalized_loss
        self.encoder_output_size = encoder_output_size

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.embed = torch.nn.Embedding(
            7 + len(self.lid_dict) + len(self.textnorm_dict), input_size
        )
        self.emo_dict = {
            "unk": 25009,
            "happy": 25001,
            "sad": 25002,
            "angry": 25003,
            "neutral": 25004,
        }

        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )
        self.beam_search = None 
        # add ctc_weight
        self.ctc_weight = 1.0 
        context_module_type = kwargs.get('context_module', '')
        if context_module_type == 'cppn':
            self.context_module = ContextModule(vocab_size,
                                        **kwargs['context_module_conf'])
        else:
            self.context_module = None
        self.bias_weight = kwargs.get('bias_weight', 1.0)

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        context_data: List[torch.Tensor],
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        
        batch_size = speech.shape[0]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)
        # 1a. Context biasing branch
        loss_bias: Optional[torch.Tensor] = None
        if self.context_module is not None:
            assert len(context_data) == 4
            context_list = context_data[0]
            context_label = context_data[1]
            context_list_lengths = context_data[2]
            context_label_lengths = context_data[3]
            context_emb = self.context_module. \
                forward_context_emb(context_list, context_list_lengths)
            encoder_out, bias_out = self.context_module(context_emb,
                                                        encoder_out)
            bias_out = bias_out.transpose(0, 1).log_softmax(2)
            loss_bias = self.context_module.bias_loss(bias_out, context_label,
                                                      encoder_out_lens,
                                                      context_label_lengths)
            loss_bias /= bias_out.size(1)
        else:
            loss_bias = None
        loss_ctc, cer_ctc = None, None
        loss_rich, acc_rich = None, None
        stats = dict()
        
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out[:, 4:, :], encoder_out_lens - 4, text[:, 4:], text_lengths - 4
        )

        loss_rich, acc_rich = self._calc_rich_ce_loss(encoder_out[:, :4, :], text[:, :4])

        loss = loss_ctc + loss_rich
        if loss is not None and loss_bias is not None:
            loss = loss + self.bias_weight * loss_bias
        # Collect total loss stats
        stats["loss_ctc"] = torch.clone(loss_ctc.detach()) if loss_ctc is not None else None
        stats["loss_rich"] = torch.clone(loss_rich.detach()) if loss_rich is not None else None
        stats["loss"] = torch.clone(loss.detach()) if loss is not None else None
        stats['loss_bias'] = torch.clone(loss_bias.detach()) if loss_bias is not None else None
        stats["acc_rich"] = acc_rich

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """

        # Data augmentation
        if self.specaug is not None and self.training:
            speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            speech, speech_lengths = self.normalize(speech, speech_lengths)

        lids = torch.LongTensor(
            [
                [
                    (
                        self.lid_int_dict[int(lid)]
                        if torch.rand(1) > 0.2 and int(lid) in self.lid_int_dict
                        else 0
                    )
                ]
                for lid in text[:, 0]
            ]
        ).to(speech.device)
        language_query = self.embed(lids)

        styles = torch.LongTensor(
            [[self.textnorm_int_dict[int(style)]] for style in text[:, 3]]
        ).to(speech.device)
        style_query = self.embed(styles)
        speech = torch.cat((style_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
       
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rich_ce_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):
        decoder_out = self.ctc.ctc_lo(encoder_out)
        # 2. Compute attention loss
        loss_rich = self.criterion_att(decoder_out, ys_pad.contiguous())
        acc_rich = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_pad.contiguous(),
            ignore_label=self.ignore_id,
        )

        return loss_rich, acc_rich
    
    def init_beam_search(
        self,
        **kwargs,
    ):
        from funasr.models.paraformer.search import BeamSearchPara
        from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        if self.ctc != None:
            ctc = CTCPrefixScorer(ctc=self.ctc, eos=self.eos, context_graph=self.context_graph)
            scorers.update(ctc=ctc)
        token_list = kwargs.get("token_list")
        #logging.info(f' token list: {token_list}')  
        scorers.update(
            length_bonus=LengthBonus(len(token_list)),
        )
        if self.context_graph and (self.ctc is None or kwargs.get("decoding_ctc_weight", 0.0) < 0.00001):
            logging.info(f'init contextscorer')
            from funasr.models.transformer.scorers.context_scorer import ContextScorer
            scorers.update(
                ctx=ContextScorer(self.context_graph)
            )
        # 3. Build ngram model
        # ngram is not supported now
        ngram = None
        scorers["ngram"] = ngram

        weights = dict(
            decoder=1.0 - kwargs.get("decoding_ctc_weight"),
            ctc=kwargs.get("decoding_ctc_weight", 0.0),
            lm=kwargs.get("lm_weight", 0.0),
            ngram=kwargs.get("ngram_weight", 0.0),
            length_bonus=kwargs.get("penalty", 0.0),
            ctx=1.0, # fix ctx weight, just adjust context_graph_score
        )
        beam_search = BeamSearchPara(
            beam_size=kwargs.get("beam_size", 2),
            weights=weights,
            scorers=scorers,
            sos=self.sos,
            eos=self.eos,
            vocab_size=len(token_list),
            token_list=token_list,
            pre_beam_score_key=kwargs.get("pre_beam_score_key", None if self.ctc_weight == 1.0 else "full"),
            pre_beam_ratio=kwargs.get("pre_beam_ratio", 1.5),
            tokenizer=kwargs.get("tokenizer", None),
            #pre_beam_score_key=None if self.ctc_weight == 1.0 else "full",
            #pre_beam_score_key='full',
        )
        # beam_search.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=kwargs.get("device", "cpu"), dtype=getattr(torch, kwargs.get("dtype", "float32"))).eval()
        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        # Initialize beam search if needed
        is_use_ctc = kwargs.get("decoding_ctc_weight", 0.0) > 0.00001 and self.ctc is not None
        is_use_lm = (
            kwargs.get("lm_weight", 0.0) > 0.00001 and kwargs.get("lm_file", None) is not None
        )
        logging.info(f"is_use_ctc: {is_use_ctc}, is_use_lm: {is_use_lm} , beam_search: {self.beam_search}") 
        logging.info(f"decoding_ctc_weight: {kwargs.get('decoding_ctc_weight')}, ctc: {self.ctc}")
        logging.info(f'context_graph_score: {kwargs.get("context_graph_score")} context_list_path: {kwargs.get("context_list_path")}')
        # init hotword_list 
        self.hotword_list = self.generate_hotwords_list(
                    kwargs.get("context_list_path", None), tokenizer=tokenizer, frontend=frontend)
        
        if self.beam_search is None and (is_use_lm or is_use_ctc):
            self.context_graph = None 
            #if kwargs.get("context_graph_score", 0.0) > 0.00001 and kwargs.get("context_list_path", None):
            if kwargs.get("context_list_path", None):
                logging.info(f'enable context_graph')
                context_list_path = kwargs.get("context_list_path")
                context_graph_score = kwargs.get("context_graph_score")
                token_list = kwargs.get("token_list", None)
                logging.info(f'token_list len: {len(token_list)}')
                symbol_table = {}
                # for i, t in enumerate(token_list):
                #     if i <= 10:
                #         logging.info(f'token_list: {t} len: {len(t)} i: {i}')
                #     symbol_table[t] = i
                # hotword
              
                logging.info(f"hotword_list: {self.hotword_list}")
                for hotword in self.hotword_list:
                    for i, t in enumerate(hotword):
                        logging.info(f'hotword_cur: {hotword} t: {t} token: {token_list[t]}')
                        symbol_table[token_list[t]] = t 
                self.context_graph = ContextGraph(self.hotword_list, symbol_table, context_graph_score, tokenizer)
            logging.info(f"enable beam_search is tokenizer SentencepiecesTokenizer: {isinstance(tokenizer, SentencepiecesTokenizer)}")
            kwargs["tokenizer"] = tokenizer
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)
        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        output_timestamp = kwargs.get("output_timestamp", False)

        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        if self.context_module is not None:
            logging.info(f'deep_biasing_score: {kwargs.get("deep_biasing_score", 1.0)} context_filtering: {kwargs.get("context_filtering", False)}')
            encoder_out = self.context_module.forward_deep_biasing(encoder_out,
                                                                    self.ctc,
                                                                    kwargs.get("deep_biasing_score", 1.0),
                                                                    self.hotword_list, 
                                                                    kwargs.get("context_filtering", False))

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")

        results = []
        b, n, d = encoder_out.size()
        if isinstance(key[0], (list, tuple)):
            key = key[0]
        if len(key) < b:
            key = key * b
        for i in range(b):
            x = encoder_out[i, : encoder_out_lens[i].item(), :]
            am_scores = ctc_logits[i, : encoder_out_lens[i].item(), :]
            #x = ctc_logits[i, : encoder_out_lens[i].item(), :]
            if self.beam_search is not None:
                if not self.beam_search.do_pre_beam:
                    am_scores = torch.zeros_like(
                        ctc_logits[i, : encoder_out_lens[i].item(), :]
                    )
                
                # Use beam search
                nbest_hyps = self.beam_search(
                    x=x,
                    am_scores=am_scores,
                    maxlenratio=kwargs.get("maxlenratio", 0.0),
                    minlenratio=kwargs.get("minlenratio", 0.0),
                )
                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                yseq = am_scores.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor([self.sos] + yseq.tolist() + [self.eos], device=yseq.device)
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
                logging.info(f'greedy search yseq: {yseq} score: {score} decode text: {tokenizer.decode(yseq.tolist())}')
            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx+1}best_recog"]
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                token_int = list(   
                    filter(
                        lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                    )
                )

                # Change integer-ids to tokens
                text = tokenizer.decode(token_int)

                # result_i = {"key": key[i], "text": text}
                # results.append(result_i)

                if ibest_writer is not None:
                    ibest_writer["text"][key[i]] = text

                if output_timestamp:
                    from itertools import groupby

                    timestamp = []
                    tokens = tokenizer.text2tokens(text)[4:]
                    token_back_to_id = tokenizer.tokens2ids(tokens)
                    token_ids = []
                    for tok_ls in token_back_to_id:
                        if tok_ls: token_ids.extend(tok_ls)
                        else: token_ids.append(124)

                    if len(token_ids) == 0:
                        result_i = {"key": key[i], "text": text}
                        results.append(result_i)
                        continue

                    logits_speech = self.ctc.softmax(encoder_out)[i, 4 : encoder_out_lens[i].item(), :]
                    pred = logits_speech.argmax(-1).cpu()
                    logits_speech[pred == self.blank_id, self.blank_id] = 0
                    align = ctc_forced_align(
                        logits_speech.unsqueeze(0).float(),
                        torch.Tensor(token_ids).unsqueeze(0).long().to(logits_speech.device),
                        (encoder_out_lens[i] - 4).long(),
                        torch.tensor(len(token_ids)).unsqueeze(0).long().to(logits_speech.device),
                        ignore_id=self.ignore_id,
                    )
                    pred = groupby(align[0, : encoder_out_lens[i]])
                    _start = 0
                    token_id = 0
                    ts_max = encoder_out_lens[i] - 4
                    for pred_token, pred_frame in pred:
                        _end = _start + len(list(pred_frame))
                        if pred_token != 0:
                            ts_left = max((_start * 60 - 30) / 1000, 0)
                            ts_right = min((_end * 60 - 30) / 1000, (ts_max * 60 - 30) / 1000)
                            timestamp.append([tokens[token_id], ts_left, ts_right])
                            token_id += 1
                        _start = _end
                    timestamp, words = self.post(timestamp)
                    result_i = {"key": key[i], "text": text, "timestamp": timestamp, "words": words}
                    results.append(result_i)
                else:
                    result_i = {"key": key[i], "text": text}
                    results.append(result_i)
        return results, meta_data

    def post(self, timestamp):
        timestamp_new = []
        words_new = []
        prev_word = None
        for i, t in enumerate(timestamp):
            word, start, end = t
            start = int(start * 1000)
            end = int(end * 1000)
            if word == "▁":
                continue
            if i == 0:
                # timestamp_new.append([word, start, end])
                timestamp_new.append([start, end])
                words_new.append(word)
            elif word.startswith("▁"):
                word = word[1:]
                timestamp_new.append([start, end])
                words_new.append(word)
            elif prev_word is not None and prev_word.isalpha() and prev_word.isascii() and word.isalpha() and word.isascii():
                prev_word += word
                timestamp_new[-1][1] = end
                words_new[-1] = prev_word
            else:
                # timestamp_new[-1][0] += word
                timestamp_new.append([start, end])
                words_new.append(word)
            prev_word = word
        return timestamp_new, words_new

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        if "max_seq_len" not in kwargs:
            kwargs["max_seq_len"] = 512
        models = export_rebuild_model(model=self, **kwargs)
        return models

        return results, meta_data

    def generate_hotwords_list(self, hotword_list_or_file, tokenizer=None, frontend=None):

        def seg_tokenize(txt, seg_dict):
            pattern = re.compile(r"^[\u4E00-\u9FA50-9]+$")
            out_txt = ""
            for word in txt:
                word = word.lower()
                if word in seg_dict:
                    out_txt += seg_dict[word] + " "
                else:
                    if pattern.match(word):
                        for char in word:
                            if char in seg_dict:
                                out_txt += seg_dict[char] + " "
                            else:
                                out_txt += "<unk>" + " "
                    else:
                        out_txt += "<unk>" + " "
            return out_txt.strip().split()

        seg_dict = None
        if frontend.cmvn_file is not None:
            model_dir = os.path.dirname(frontend.cmvn_file)
            seg_dict_file = os.path.join(model_dir, "seg_dict")
            if os.path.exists(seg_dict_file):
                seg_dict = load_seg_dict(seg_dict_file)
            else:
                seg_dict = None
        # for None
        if hotword_list_or_file is None:
            hotword_list = None
        # for local txt inputs
        elif os.path.exists(hotword_list_or_file) and hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords from local txt...")
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    res = tokenizer.tokens2ids(hw_list)
                    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
                        res = res[0]
                    hotword_list.append(res)
                #hotword_list.append([self.sos])
                #hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for url, download and generate txt
        elif hotword_list_or_file.startswith("http"):
            logging.info("Attempting to parse hotwords from url...")
            work_dir = tempfile.TemporaryDirectory().name
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            text_file_path = os.path.join(work_dir, os.path.basename(hotword_list_or_file))
            local_file = requests.get(hotword_list_or_file)
            open(text_file_path, "wb").write(local_file.content)
            hotword_list_or_file = text_file_path
            hotword_list = []
            hotword_str_list = []
            with codecs.open(hotword_list_or_file, "r") as fin:
                for line in fin.readlines():
                    hw = line.strip()
                    hw_list = hw.split()
                    if seg_dict is not None:
                        hw_list = seg_tokenize(hw_list, seg_dict)
                    hotword_str_list.append(hw)
                    hotword_list.append(tokenizer.tokens2ids(hw_list))
                hotword_list.append([self.sos])
                hotword_str_list.append("<s>")
            logging.info(
                "Initialized hotword list from file: {}, hotword list: {}.".format(
                    hotword_list_or_file, hotword_str_list
                )
            )
        # for text str input
        elif not hotword_list_or_file.endswith(".txt"):
            logging.info("Attempting to parse hotwords as str...")
            hotword_list = []
            hotword_str_list = []
            for hw in hotword_list_or_file.strip().split():
                hotword_str_list.append(hw)
                hw_list = hw.strip().split()
                if seg_dict is not None:
                    hw_list = seg_tokenize(hw_list, seg_dict)
                hotword_list.append(tokenizer.tokens2ids(hw_list))
            hotword_list.append([self.sos])
            hotword_str_list.append("<s>")
            logging.info("Hotword list: {}.".format(hotword_str_list))
        else:
            hotword_list = None
        return hotword_list

@lru_cache(maxsize=1)
def load_seg_dict(seg_dict_file):
    seg_dict = {}
    assert isinstance(seg_dict_file, str)
    with open(seg_dict_file, "r", encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            s = line.strip().split()
            key = s[0]
            value = s[1:]
            seg_dict[key] = " ".join(value)
    return seg_dict
