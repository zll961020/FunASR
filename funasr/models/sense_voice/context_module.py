# Copyright (c) 2023 ASLP@NWPU (authors: Kaixun Huang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import torch
import torch.nn as nn
from typing import Tuple
#from wenet.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.attention import MultiHeadedAttention
from torch.nn.utils.rnn import pad_sequence
from funasr.models.ctc.ctc import CTC
from typing import Dict, List

class BLSTM(torch.nn.Module):
    """Context encoder, encoding unequal-length context phrases
       into equal-length embedding representations.
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 num_layers,
                 dropout=0.0):
        super(BLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(
            self.vocab_size, self.embedding_size)

        self.sen_rnn = torch.nn.LSTM(input_size=self.embedding_size,
                                     hidden_size=self.embedding_size,
                                     num_layers=num_layers,
                                     dropout=dropout,
                                     batch_first=True,
                                     bidirectional=True)

    def forward(self, sen_batch, sen_lengths):
        sen_batch = torch.clamp(sen_batch, 0)
        sen_batch = self.word_embedding(sen_batch)
        pack_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sen_batch, sen_lengths.to('cpu').type(torch.int32),
            batch_first=True, enforce_sorted=False)
        _, last_state = self.sen_rnn(pack_seq)
        laste_h = last_state[0]
        laste_c = last_state[1]
        state = torch.cat([laste_h[-1, :, :], laste_h[-2, :, :],
                          laste_c[-1, :, :], laste_c[-2, :, :]], dim=-1)
        return state


class ContextModule(torch.nn.Module):
    """Context module, Using context information for deep contextual bias

    During the training process, the original parameters of the ASR model
    are frozen, and only the parameters of context module are trained.

    Args:
        vocab_size (int): vocabulary size
        embedding_size (int): number of ASR encoder projection units
        encoder_layers (int): number of context encoder layers
        attention_heads (int): number of heads in the biasing layer
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.vocab_size = vocab_size
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate

        self.context_extractor = BLSTM(self.vocab_size, self.embedding_size,
                                       self.encoder_layers)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.embedding_size * 4, self.embedding_size),
            nn.LayerNorm(self.embedding_size)
        )

        self.biasing_layer = MultiHeadedAttention(
            n_head=self.attention_heads,
            n_feat=self.embedding_size,
            dropout_rate=self.dropout_rate
        )

        self.combiner = nn.Linear(self.embedding_size, self.embedding_size)
        self.norm_aft_combiner = nn.LayerNorm(self.embedding_size)

        self.context_decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.context_decoder_ctc_linear = nn.Linear(self.embedding_size,
                                                    self.vocab_size)

        self.bias_loss = torch.nn.CTCLoss(reduction="sum", zero_infinity=True)

    def forward_context_emb(self,
                            context_list: torch.Tensor,
                            context_lengths: torch.Tensor
                            ) -> torch.Tensor:
        """Extracting context embeddings
        """
        context_emb = self.context_extractor(context_list, context_lengths)
        context_emb = self.context_encoder(context_emb.unsqueeze(0))
        return context_emb

    def forward(self,
                context_emb: torch.Tensor,
                encoder_out: torch.Tensor,
                biasing_score: float = 1.0,
                recognize: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Using context embeddings for deep biasing.

        Args:
            biasing_score (int): degree of context biasing
            recognize (bool): no context decoder computation if True
        """
        context_emb = context_emb.expand(encoder_out.shape[0], -1, -1)
        context_emb = self.biasing_layer(encoder_out, context_emb,
                                            context_emb, None)
        encoder_bias_out = \
            self.norm_aft_combiner(encoder_out +
                                   self.combiner(context_emb) * biasing_score)
        if recognize:
            return encoder_bias_out, torch.tensor(0.0)
        bias_out = self.context_decoder(context_emb)
        bias_out = self.context_decoder_ctc_linear(bias_out)
        return encoder_bias_out, bias_out

    def get_context_list_tensor(self, context_list: List[List[int]]):
        """Add 0 as no-bias in the context list and obtain the tensor
           form of the context list
        """
        context_list_tensor = [torch.tensor([0], dtype=torch.int32)]
        for context_token in context_list:
            context_list_tensor.append(torch.tensor(context_token, dtype=torch.int32))
        context_list_lengths = torch.tensor([x.size(0) for x in context_list_tensor],
                                            dtype=torch.int32)
        context_list_tensor = pad_sequence(context_list_tensor,
                                           batch_first=True,
                                           padding_value=-1)
        return context_list_tensor, context_list_lengths
    
    def forward_deep_biasing(self,
                             encoder_out: torch.Tensor,
                             ctc: CTC,
                             deep_biasing_score: float, 
                             hotword_list: List[List[int]], context_filtering=False):
        """Apply deep biasing based on encoder output and context list
        """
        if context_filtering:
            ctc_probs = ctc.log_softmax(encoder_out).squeeze(0)
            filtered_context_list = self.two_stage_filtering(
                hotword_list, ctc_probs)
            context_list, context_list_lengths = self. \
                get_context_list_tensor(filtered_context_list)
        else:
            context_list, context_list_lengths = self. \
                get_context_list_tensor(hotword_list)
        context_list = context_list.to(encoder_out.device)
        context_emb = self.forward_context_emb(context_list, context_list_lengths)
        encoder_out, _ = self.forward(context_emb, encoder_out,
                           deep_biasing_score, True)
        return encoder_out

    def two_stage_filtering(self,
                            context_list: List[List[int]],
                            ctc_posterior: torch.Tensor,
                            filter_window_size: int = 64):
        """Calculate PSC and SOC for context phrase filtering,
           refer to: https://arxiv.org/abs/2301.06735
        """
        if len(context_list) == 0:
            return context_list

        SOC_score = {}
        for t in range(1, ctc_posterior.shape[0]):
            if t % (filter_window_size // 2) != 0 and t != ctc_posterior.shape[0] - 1:
                continue
            # calculate PSC
            PSC_score = {}
            max_posterior, _ = torch.max(ctc_posterior[max(0,
                                         t - filter_window_size):t, :],
                                         dim=0, keepdim=False)
            max_posterior = max_posterior.tolist()
            for i in range(len(context_list)):
                score = sum(max_posterior[j] for j in context_list[i]) \
                    / len(context_list[i])
                PSC_score[i] = max(SOC_score.get(i, -float('inf')), score)
            PSC_filtered_index = []
            for i in PSC_score:
                if PSC_score[i] > self.filter_threshold:
                    PSC_filtered_index.append(i)
            if len(PSC_filtered_index) == 0:
                continue
            filtered_context_list = []
            for i in PSC_filtered_index:
                filtered_context_list.append(context_list[i])

            # calculate SOC
            win_posterior = ctc_posterior[max(0, t - filter_window_size):t, :]
            win_posterior = win_posterior.unsqueeze(0) \
                .expand(len(filtered_context_list), -1, -1)
            select_win_posterior = []
            for i in range(len(filtered_context_list)):
                select_win_posterior.append(torch.index_select(
                    win_posterior[0], 1,
                    torch.tensor(filtered_context_list[i],
                                 device=ctc_posterior.device)).transpose(0, 1))
            select_win_posterior = \
                pad_sequence(select_win_posterior,
                             batch_first=True).transpose(1, 2).contiguous()
            dp = torch.full((select_win_posterior.shape[0],
                             select_win_posterior.shape[2]),
                            -10000.0, dtype=torch.float32,
                            device=select_win_posterior.device)
            dp[:, 0] = select_win_posterior[:, 0, 0]
            for win_t in range(1, select_win_posterior.shape[1]):
                temp = dp[:, :-1] + select_win_posterior[:, win_t, 1:]
                idx = torch.where(temp > dp[:, 1:])
                idx_ = (idx[0], idx[1] + 1)
                dp[idx_] = temp[idx]
                dp[:, 0] = \
                    torch.where(select_win_posterior[:, win_t, 0] > dp[:, 0],
                                select_win_posterior[:, win_t, 0], dp[:, 0])
            for i in range(len(filtered_context_list)):
                SOC_score[PSC_filtered_index[i]] = \
                    max(SOC_score.get(PSC_filtered_index[i], -float('inf')),
                        dp[i][len(filtered_context_list[i]) - 1]
                        / len(filtered_context_list[i]))
        filtered_context_list = []
        for i in range(len(context_list)):
            if SOC_score.get(i, -float('inf')) > self.filter_threshold:
                filtered_context_list.append(context_list[i])
        return filtered_context_list

