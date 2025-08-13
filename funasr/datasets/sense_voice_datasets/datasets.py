import logging

import re
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from torch.nn.utils.rnn import pad_sequence


@tables.register("dataset_classes", "SenseVoiceDataset")
class SenseVoiceDataset(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 5)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):

        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            source = item["source"]
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)
            speech, speech_lengths = extract_fbank(
                data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
            )  # speech: [b, T, d]

            if speech_lengths > self.batch_size:
                continue
            if self.permute:
                speech = speech.permute(0, 2, 1)
            target = item["target"]
            if self.preprocessor_text:
                target = self.preprocessor_text(target)

            task = item.get("prompt", "<|ASR|>")
            text_language = item.get("text_language", "<|zh|>")

            if isinstance(self.sos, str):
                prompt = f"{self.sos}{task}{text_language}"
                prompt_ids = self.tokenizer.encode(prompt, allowed_special="all")
            else:
                prompt = f"{task}{text_language}"
                prompt_ids = self.tokenizer.encode(prompt, allowed_special="all")
                prompt_ids = [self.sos] + prompt_ids

            prompt_ids_len = len(prompt_ids) - 1  # [sos, task]
            self.prompt_ids_len = prompt_ids_len

            target_ids = self.tokenizer.encode(target, allowed_special="all")
            target_ids_len = len(target_ids) + 1  # [lid, text]
            if target_ids_len > 200:
                continue

            if isinstance(self.eos, str):
                eos = self.tokenizer.encode(self.eos, allowed_special="all")  # [eos]
            else:
                eos = [self.eos]

            ids = prompt_ids + target_ids + eos  # [sos, task, lid, text, eos]
            ids_lengths = len(ids)

            text = torch.tensor(ids, dtype=torch.int64)
            text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

            target_mask = (
                [0] * (prompt_ids_len) + [1] * (target_ids_len) + [1]
            )  # [sos, task, lid, text, eos]: [0, 0, 1, 1, 1]
            target_mask_lengths = len(target_mask)
            target_mask = torch.tensor(target_mask, dtype=torch.float32)
            target_mask_lengths = torch.tensor([target_mask_lengths], dtype=torch.int32)

            output = {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
                "target_mask": target_mask,
                "target_mask_lengths": target_mask_lengths,
            }
            break

        return output

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
            if sample is None:
                continue
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        if len(outputs) < 1:
            logging.error(f"ERROR: data is empty!")
            outputs = {
                "speech": torch.rand((10, 128), dtype=torch.float32)[None, :, :],
                "speech_lengths": torch.tensor(
                    [
                        10,
                    ],
                    dtype=torch.int32,
                )[:, None],
                "text": torch.tensor(
                    [
                        58836,
                    ],
                    dtype=torch.int32,
                )[None, :],
                "text_lengths": torch.tensor(
                    [
                        1,
                    ],
                    dtype=torch.int32,
                )[:, None],
                "target_mask": torch.tensor([[0] * (self.prompt_ids_len) + [1] * (1) + [1]])[
                    None, :
                ],
            }
            return outputs

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

        if self.batch_type != "example":
            for i in range(10):
                outputs = self._filter_badcase(outputs, i=i)

        return outputs

    def _filter_badcase(self, outputs, i=0):
        b, t, _ = outputs["speech"].shape

        if b * t > self.batch_size * 1.25:
            beg = torch.randint(0, 2, ()).item()
            if b < 2:
                beg = 0
            logging.info(
                f"Warning, b * t: {b * t} > {self.batch_size}, drop half data {i}th, beg:{beg}"
            )
            for key, data_list in outputs.items():
                outputs[key] = outputs[key][beg : beg + b : 2]

            speech_lengths_max = outputs["speech_lengths"].max().item()
            outputs["speech"] = outputs["speech"][:, :speech_lengths_max, :]
            text_lengths_max = outputs["text_lengths"].max().item()
            outputs["text"] = outputs["text"][:, :text_lengths_max]
            target_mask_lengths_max = outputs["target_mask_lengths"].max().item()
            outputs["target_mask"] = outputs["target_mask"][:, :target_mask_lengths_max]

        return outputs


@tables.register("dataset_classes", "SenseVoiceCTCDataset")
class SenseVoiceCTCDataset(torch.utils.data.Dataset):
    """
    SenseVoiceCTCDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 5)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):

        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            source = item["source"]
            try:
                data_src = load_audio_text_image_video(source, fs=self.fs)
            except Exception as e:
                logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")
                continue

            if self.preprocessor_speech:
                data_src = self.preprocessor_speech(data_src, fs=self.fs)
            speech, speech_lengths = extract_fbank(
                data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
            )  # speech: [b, T, d]

            if speech_lengths > self.batch_size:
                continue
            if self.permute:
                speech = speech.permute(0, 2, 1)
            asr_target = item["target"]
            if self.preprocessor_text:
                asr_target = self.preprocessor_text(asr_target)
            emo_target = item.get("emo_target", "<|NEUTRAL|>")
            event_target = item.get("event_target", "<|Speech|>")
            text_language = item.get("text_language", "<|zh|>")
            punc_itn_bottom = item.get("with_or_wo_itn", "<|woitn|>")

            target_ids = self.tokenizer.encode(asr_target, allowed_special="all")
            target_ids_len = len(target_ids)  # [text]
            if target_ids_len > 200:
                continue

            lid_ids = self.tokenizer.encode(text_language, allowed_special="all")
            emo_ids = self.tokenizer.encode(emo_target, allowed_special="all")
            event_ids = self.tokenizer.encode(event_target, allowed_special="all")
            punc_itn_bottom_ids = self.tokenizer.encode(punc_itn_bottom, allowed_special="all")

            ids = lid_ids + emo_ids + event_ids + punc_itn_bottom_ids + target_ids # [lid, emo, lid, itn, text]
            ids_lengths = len(ids)

            text = torch.tensor(ids, dtype=torch.int64)
            text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

            output = {
                "speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
            }
            break

        return output

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
            if sample is None:
                continue
            for key in sample.keys():
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(sample[key])

        if len(outputs) < 1:
            logging.error(f"ERROR: data is empty!")
            outputs = {
                "speech": torch.rand((10, 128), dtype=torch.float32)[None, :, :],
                "speech_lengths": torch.tensor(
                    [
                        10,
                    ],
                    dtype=torch.int32,
                )[:, None],
                "text": torch.tensor(
                    [
                        58836,
                    ],
                    dtype=torch.int32,
                )[None, :],
                "text_lengths": torch.tensor(
                    [
                        1,
                    ],
                    dtype=torch.int32,
                )[:, None],
            }
            return outputs

        for key, data_list in outputs.items():
            if isinstance(data_list[0], torch.Tensor):
                if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                    pad_value = self.int_pad_value
                else:
                    pad_value = self.float_pad_value

                outputs[key] = torch.nn.utils.rnn.pad_sequence(
                    data_list, batch_first=True, padding_value=pad_value
                )

        if self.batch_type != "example":
            for i in range(10):
                outputs = self._filter_badcase(outputs, i=i)

        return outputs

    def _filter_badcase(self, outputs, i=0):
        b, t, _ = outputs["speech"].shape

        if b * t > self.batch_size * 1.25:
            beg = torch.randint(0, 2, ()).item()
            if b < 2:
                beg = 0
            logging.info(
                f"Warning, b * t: {b * t} > {self.batch_size}, drop half data {i}th, beg:{beg}"
            )
            for key, data_list in outputs.items():
                outputs[key] = outputs[key][beg : beg + b : 2]

            speech_lengths_max = outputs["speech_lengths"].max().item()
            outputs["speech"] = outputs["speech"][:, :speech_lengths_max, :]
            text_lengths_max = outputs["text_lengths"].max().item()
            outputs["text"] = outputs["text"][:, :text_lengths_max]

        return outputs

@tables.register("dataset_classes", "SenseVoiceCTCDatasetHotword")
class SenseVoiceCTCDatasetHotword(SenseVoiceCTCDataset):
    """
    Same as SenseVoiceCTCDataset but additionally provides:
        hotword_pad      – (N_hotword+1, max_len)
        hotword_lengths  – (N_hotword+1,)
    so that models that need contextual-bias (hotword) inputs can
    be trained without修改 AudioDatasetHotword 逻辑。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # ---------------- Sample -----------------
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        length = sample["text_lengths"][0].item()
        offset = 4                  # 固定跳过 4 个 tag
        content_len = max(0, length - offset)
        
        # ------- Hotword span sampler (与 AudioDatasetHotword 等价) --------
        def generate_index(
                length,
                hotword_min_length=2,
                hotword_max_length=8,
                sample_rate=0.75,
                double_rate=0.1,
                pre_prob=0.0,
                pre_index=None,
                pre_hwlist=None,
            ):
            if length < hotword_min_length:
                return [-1]
            if random.random() < sample_rate:
                if pre_prob > 0 and random.random() < pre_prob and pre_index is not None:
                    return pre_index
                if length == hotword_min_length:
                    return [0, length - 1]
                elif (
                    random.random() < double_rate
                    and length > hotword_max_length + hotword_min_length + 2
                ):
                    # sample two hotwords in a sentence
                    _max_hw_length = min(hotword_max_length, length // 2)
                    # first hotword
                    start1 = random.randint(0, length // 3)
                    end1 = random.randint(
                        start1 + hotword_min_length - 1, start1 + _max_hw_length - 1
                    )
                    # second hotword
                    start2 = random.randint(end1 + 1, length - hotword_min_length)
                    end2 = random.randint(
                        min(length - 1, start2 + hotword_min_length - 1),
                        min(length - 1, start2 + hotword_max_length - 1),
                    )
                    return [start1, end1, start2, end2]
                else:  # single hotword
                    start = random.randint(0, length - hotword_min_length)
                    end = random.randint(
                        min(length - 1, start + hotword_min_length - 1),
                        min(length - 1, start + hotword_max_length - 1),
                    )
                    return [start, end]
            else:
                return [-1]

        hw_idx = generate_index(content_len)

        # 将相对位置平移到真实位置
        if hw_idx[0] != -1:
            hw_idx = [i + offset for i in hw_idx]

        sample["hotword_indx"] = hw_idx
        return sample

    # ---------------- Batch -----------------
    def collator(self, samples):
        base_out = super().collator(samples)

        hotword_indxs = [s["hotword_indx"] for s in samples]
        hotword_list, hotword_lengths = [], []
        text_tensor = base_out["text"]

        for idx, (hw_idx, one_text, tl) in enumerate(
            zip(hotword_indxs, text_tensor, base_out["text_lengths"])
        ):
            tl = tl[0]
            if hw_idx[0] != -1:
                s, e = hw_idx[0], hw_idx[1]
                hotword_list.append(one_text[s : e + 1])
                hotword_lengths.append(e - s + 1)
                if len(hw_idx) == 4 and hw_idx[2] != -1:
                    s, e = hw_idx[2], hw_idx[3]
                    hotword_list.append(one_text[s : e + 1])
                    hotword_lengths.append(e - s + 1)

        # 追加占位 <s>=1，防止 batch 内无热词
        hotword_list.append(torch.tensor([1]))
        hotword_lengths.append(1)

        hotword_pad = torch.nn.utils.rnn.pad_sequence(
            hotword_list, batch_first=True, padding_value=0
        )
        base_out["hotword_pad"] = hotword_pad
        base_out["hotword_lengths"] = torch.tensor(hotword_lengths, dtype=torch.int32)

        return base_out

@tables.register("dataset_classes", "SenseVoiceCTCDatasetHotwordCPNN")
class SenseVoiceCTCDatasetHotwordCPNN(SenseVoiceCTCDataset):
    """在 Hotword 数据集基础上，额外产出 CPNN 所需的 context_data"""

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # ======== CPNN 相关超参 ========
        context_conf = kwargs.get("context_conf", None)
        self.len_min = context_conf.get("len_min", 2)
        self.len_max = context_conf.get("len_max", 8)
        self.utt_num_context = context_conf.get("utt_num_context", 3)
        self.batch_num_context = context_conf.get("batch_num_context", 60)

        # ======== 构造 symbol_table ↔︎ id 的映射，供 context_sampling 使用 ========
        # SentencepiecesTokenizer:  self.tokenizer.sp  -> SentencePieceProcessor
        token_list = kwargs.get("token_list", None)
        logging.info(f'token_list len: {len(token_list)}')
        self.symbol_table = {}
        for i, t in enumerate(token_list):
            if i <= 10:
                logging.info(f'token_list: {t} len: {len(t)} i: {i}')
            self.symbol_table[t] = i
        # sp = getattr(self.tokenizer, "sp", None)
        # if sp is None:
        #     raise ValueError("tokenizer must be SentencepiecesTokenizer")

        # self.symbol_table = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
    # def __getitem__(self, index):
    #     sample = super().__getitem__(index)
    #     # 取出单条样本的有效长度 L
    #     l = int(sample["text_lengths"].item())
    #     lab = sample["text"][:l].clone()

    #     # 构造单样本 stub，并调用 context_sampling
    #     sample_stub = [{"label": lab}]
    #     _ = list(context_sampling([sample_stub],
    #                                 self.symbol_table,
    #                                 self.len_min, self.len_max,
    #                                 self.utt_num_context,
    #                                 self.batch_num_context,
    #                                 skip_token=4))
    #     sample["context_list"] = sample_stub[0]["context_list"]
    #     return sample

    # ------------- Batch -------------
    def collator(self, samples: list):
        """
        返回:
            {
              speech, speech_lengths,
              text,   text_lengths,
              hotword_pad, hotword_lengths,   # ← 来自父类
              context_data=[pad_ctx_list, pad_ctx_label,
                            ctx_list_lens, ctx_label_lens]  # ← 新增
            }
        """
        # 1. 先调用父类整理出基础张量
        base_out = super().collator(samples)

        # 2. 取出去 padding 的 label (List[Tensor])
        b_text      = base_out["text"]
        b_text_len  = base_out["text_lengths"].squeeze(1)
        batch_label = [b_text[i, :l].clone() for i, l in enumerate(b_text_len)]

        # 3. 用 context_sampling 生成 batch 级的 context_list
        sample_stub = [{"label": lab} for lab in batch_label]  # dummy list
        # context_sampling 期望的参数: Iterable[List[dict]], 所以再套一层 list
        _ = list(context_sampling([sample_stub],    # data
                                  self.symbol_table,
                                  self.len_min, self.len_max,
                                  self.utt_num_context,
                                  self.batch_num_context,
                                  skip_token=4))
        context_list = sample_stub[0]["context_list"]  # List[Tensor]

        # 4. 依据 context_list 反向生成每条语句的 context_label
        context_labels = context_label_generate(batch_label, context_list, skip_token=4)

        # ---------- Padding ----------
        context_list_lengths = torch.tensor([x.size(0) for x in context_list],
                                            dtype=torch.int32)
        pad_ctx_list = pad_sequence(
            context_list, batch_first=True, padding_value=-1
        )

        context_label_lengths = torch.tensor(
            [x.size(0) for x in context_labels], dtype=torch.int32
        )
        pad_ctx_label = pad_sequence(
            context_labels, batch_first=True, padding_value=-1
        )

        # 5. 打包
        base_out["context_data"] = [
            pad_ctx_list,
            pad_ctx_label,
            context_list_lengths,
            context_label_lengths,
        ]
        return base_out


####################  新增：CPNN 相关辅助函数 ####################
def context_sampling(
    data,
    symbol_table,
    len_min,
    len_max,
    utt_num_context,
    batch_num_context,
    skip_token: int = 0,          # ← 跳过前 N 个 token
):
    """
    随机从句子中截取若干片段作为热词，得到 batch 级的 context_list
    """
    # id -> piece 的反查表
    rev_symbol_table = {v: k for k, v in symbol_table.items()}

    context_list_over_all = []
    for sample in data:
        batch_label = [s["label"] for s in sample]
        context_list = []

        for utt_label in batch_label:
            # -------- 1. 句首控制符位置跳过 --------
            st_index_list = []
            for i in range(skip_token, len(utt_label)):
                # 仅在 sentencepiece 的“▁”处作为 token 起始
                if "▁" not in symbol_table:
                    st_index_list.append(i)
                elif rev_symbol_table[int(utt_label[i])][0] == "▁":
                    st_index_list.append(i)
                else:
                    st_index_list.append(i)
            st_index_list.append(len(utt_label))

            # -------- 2. 随机采样若干 [start, end) 片段 --------
            st_select, en_select = [], []
            for _ in range(utt_num_context):
                random_len = random.randint(
                    min(len(st_index_list) - 1, len_min),
                    min(len(st_index_list) - 1, len_max),
                )
                random_index = random.randint(0, len(st_index_list) - random_len - 1)
                st_idx = st_index_list[random_index]
                en_idx = st_index_list[random_index + random_len]

                # 与已选片段不交叉
                cross = any(
                    (st_idx >= s and st_idx < e)
                    or (en_idx > s and en_idx <= e)
                    or (st_idx < s and en_idx > e)
                    for s, e in zip(st_select, en_select)
                )
                if not cross:
                    context_list.append(utt_label[st_idx:en_idx])
                    st_select.append(st_idx)
                    en_select.append(en_idx)

        # -------- 3. 控制 batch 级 context 总数 --------
        if len(context_list) > batch_num_context:
            context_list_over_all = context_list
        elif len(context_list) + len(context_list_over_all) > batch_num_context:
            context_list_over_all.extend(context_list)
            context_list_over_all = context_list_over_all[-batch_num_context:]
        else:
            context_list_over_all.extend(context_list)

        # 0 作为 no-bias 占位
        context_list = context_list_over_all.copy()
        context_list.insert(0, torch.tensor([0], dtype=torch.int32))

        # List[int] → Tensor
        context_list = [
            torch.tensor(c, dtype=torch.int32) if not isinstance(c, torch.Tensor) else c
            for c in context_list
        ]
        sample[0]["context_list"] = context_list
        yield sample


def context_label_generate(label, context_list, skip_token: int = 0):
    """
    根据 context_list 生成与每条语句对齐的 context_label
    """
    context_labels = []
    for x in label:
        cur_len = len(x)
        context_label, cnt = [], 0
        # 跳过前 skip_token 个控制符
        for i in range(skip_token, cur_len):
            for j in range(1, len(context_list)):
                if i + len(context_list[j]) > cur_len:
                    continue
                if x[i : i + len(context_list[j])].equal(context_list[j]):
                    cnt = max(cnt, len(context_list[j]))
            if cnt > 0:
                context_label.append(x[i])
                cnt -= 1
        context_labels.append(torch.tensor(context_label, dtype=torch.int64))
    return context_labels
###################################################################