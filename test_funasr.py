
import logging
# 在导入 FunASR 之前设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # 强制重新配置，覆盖后续的配置
)
from funasr.models.seaco_paraformer.model import SeacoParaformer
from funasr.models.sense_voice.model import SenseVoiceSmall, SenseVoiceSmallCPNN
from funasr import AutoModel 
import json 
import re, string
import json  
from pathlib import Path
import argparse
import os 
import torch  
emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	# s_list[0] 可能为空，为了后续 ^ 匹配方便加一个空格前缀
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()

	# ---------- 去掉可能残留的控制符前缀 ----------
	# 允许前面带空格；只匹配一次
	ctrl_prefix_pat = re.compile(
		r'^\s*'                                 # 开头可有空白
		r'(?:zh|en|yue|ja|ko|nospeech)?'        # 语言
		r'(?:HAPPY|SAD|ANGRY|NEUTRAL|FEARFUL|DISGUSTED|SURPRISED)?'  # 情感
		r'(?:Speech)?'                          # Speech
		r'(?:woitn|withitn)?'                   # ITN 标记
	)
	new_s = re.sub(ctrl_prefix_pat, '', new_s, count=1).lstrip()
	# -----------------------------------------------------
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def strip_punctuation(text: str) -> str:
    # 英文 + 常见中日韩符号，如需更多自行补充
    punctuation = string.punctuation + "，。！？；：【】（）《》‘’“”…—、"
    return re.sub(f"[{re.escape(punctuation)}]", "", text)

def transcribe(model, audio_path, batch_size, hotword_list_txt=None, context_graph_score=10.0, decoding_ctc_weight=0.4, beam_size=5, mode='greedy_search', 
               lm_weight=0.0, lm_file=None, pre_beam_score_key=None, pre_beam_ratio=1.5, deep_biasing_score=1.0):
    if mode == 'greedy_search':
        logging.info('init greedy_search')
        text = model.generate(audio_path, batch_size=batch_size, hotword=hotword_list_txt, context_list_path=hotword_list_txt, deep_biasing_score=deep_biasing_score)[0]['text']
    elif mode == 'beam_search':
        logging.info(f"decoding_ctc_weight: {decoding_ctc_weight}, beam_size: {beam_size}") 
        text = model.generate(audio_path, batch_size=batch_size, hotword=hotword_list_txt, decoding_ctc_weight=decoding_ctc_weight,
                              beam_size=beam_size, context_list_path=hotword_list_txt,
                                context_graph_score=context_graph_score, lm_weight=lm_weight, lm_file=lm_file,
                                pre_beam_score_key=pre_beam_score_key, pre_beam_ratio=pre_beam_ratio)[0]['text']
    
    return text 

# ---------- 主逻辑 ----------
def main(args):
   
    """批量转录文本 + 断点续跑"""
    # ------------ 构建模型（一次） ------------
    model = AutoModel(model=args.model,
                      device="cuda:0",
                      disable_update=True,
                      disable_log=True)   # 关掉网络版本检查
    # ------------ 恢复机制 ------------
    processed_keys = set()
    if Path(args.output).is_file():
        with open(args.output, "r", encoding="utf-8") as f_exist:
            for line in f_exist:
                line = line.strip()
                if not line:
                    continue
                # 输出格式:  key<space>text
                processed_keys.add(line.split(maxsplit=1)[0])
        print(f"[Resume] 已读取 {len(processed_keys)} 条已完成数据，运行时将跳过。")

    # 统计
    total   = len(processed_keys) 
    ok      = len(processed_keys)   # 已完成也算成功
    fail    = 0

    # 根据是否恢复决定写文件模式
    file_mode = "a" if processed_keys else "w"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ## 初始化homophonereplacer
    if args.replace_fst:
        from sherpa_onnx import HomophoneReplacer, HomophoneReplacerConfig

        cfg = HomophoneReplacerConfig(
            dict_dir="hr_resources/dict",   # 没有分词需求可设成 ""
            lexicon="hr_resources/lexicon.txt",   # 必填
            rule_fsts=args.replace_fst,                         # 若有规则 FST，填路径
            debug=True                            # 打开日志，观察处理过程
        )

        hr = HomophoneReplacer(cfg)

    # 打开文件
    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, file_mode, encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line.strip())
            key = item["key"]

            # ---- 已处理则跳过 ----
            if key in processed_keys:
                if args.verbose:
                    print(f"[Skip] {key} 已存在，跳过。")
                continue

            total += 1
            try:
                wav_path = item["wav"]
                if not Path(wav_path).is_file():
                    raise FileNotFoundError(wav_path)

                # 1. 先得到原始转写
                text_raw = transcribe(
                    model, wav_path, args.batch_size,
                    args.hotwords, args.context_graph_score,
                    args.decoding_ctc_weight, args.beam_size, args.mode,
                    args.lm_weight, args.lm_file, args.pre_beam_score_key,
                    args.pre_beam_ratio,
                    args.deep_biasing_score, 
                )
                # 2. 如果是 SenseVoiceSmall，先做特殊标记清理
                if isinstance(model.model, SenseVoiceSmall) or isinstance(model.model, SenseVoiceSmallCPNN):
                    text_raw = format_str_v3(text_raw)
                # 3. 最后再去掉标点
                text = strip_punctuation(text_raw)
                text = text.replace(" ", "")
                if args.replace_fst:
                    text = hr.apply(text)

                fout.write(f"{key} {text}\n")
                ok += 1
                if args.verbose:
                    print(f"[{ok}/{total}] {key}: {text}")
            except Exception as e:
                fail += 1
                print(f"[ERROR] line {total}: {e}")
            # 及时清显存，防止 CUDA 内存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. success={ok}, fail={fail}, total={total}")
    print(f"结果已写入: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="wav 列表 txt（JSON-Lines）")
    parser.add_argument("--output", type=str, help="输出 txt")
    parser.add_argument("--model", type=str, help="模型路径")
    parser.add_argument("--hotwords", type=str, default=None, help="热词列表 txt")
    parser.add_argument("--context_graph_score", type=float, default=10.0, help="上下文图得分")
    parser.add_argument("--decoding_ctc_weight", type=float, default=0.4, help="CTC 权重")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--mode", type=str, default='greedy_search', help="搜索模式")
    parser.add_argument("--beam_size", type=int, default=5, help="beam 大小")
    parser.add_argument("--lm_weight", type=float, default=0.0, help="LM 权重")
    parser.add_argument("--lm_file", type=str, default=None, help="LM 文件")
    parser.add_argument("--pre_beam_score_key", type=str, default=None, help="pre beam score key")
    parser.add_argument("--pre_beam_ratio", type=float, default=1.5, help="pre beam ratio")
    parser.add_argument("--verbose", action="store_true", help="打印逐条结果")
    parser.add_argument("--replace_fst", type=str, default=None, help="替换 fst 文件")
    parser.add_argument("--deep_biasing_score", type=float, default=1.0, help="deep biasing 得分")

    args = parser.parse_args()
    main(args)