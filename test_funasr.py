from funasr import AutoModel 
import json 
import re, string
import json  
from pathlib import Path
import argparse
import os 
import torch  

def strip_punctuation(text: str) -> str:
    # 英文 + 常见中日韩符号，如需更多自行补充
    punctuation = string.punctuation + "，。！？；：【】（）《》‘’“”…—、"
    return re.sub(f"[{re.escape(punctuation)}]", "", text)

def transcribe(model, audio_path, batch_size,hotword_list_txt=None):
    text = model.generate(audio_path, batch_size=batch_size, hotword=hotword_list_txt)[0]['text']
    return text 

# ---------- 主逻辑 ----------
def main(args):
   
    """批量转录文本 + 断点续跑"""
    # ------------ 构建模型（一次） ------------
    model = AutoModel(model=args.model,
                      device="cuda:0",
                      disable_update=True)   # 关掉网络版本检查
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

                text = strip_punctuation(
                    transcribe(model, wav_path, args.batch_size, args.hotwords)
                )
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
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--verbose", action="store_true", help="打印逐条结果")
    args = parser.parse_args()
    main(args)