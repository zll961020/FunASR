#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage
-----
python gen_ocr_list.py \
    --hotwords hotwords.txt \
    --ref token.ref \
    --out ocr.list
"""
import argparse
import io
import re
from pathlib import Path

def load_hotwords(path: Path):
    """返回热词列表（保持原顺序，去掉空行/首尾空格）"""
    words = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w and w not in words:
                words.append(w)
    return words

def build_regex(words):
    """
    将热词列表编成一个 OR 正则，以便一次性搜索。
    对特殊字符做转义；加上 (?u) 让 re 在 unicode 环境下工作。
    """
    escaped = [re.escape(w) for w in words]
    pat = "(?u)(" + "|".join(escaped) + ")"
    return re.compile(pat)

def main(args):
    hotwords = load_hotwords(Path(args.hotwords))
    if len(hotwords) == 0:
        raise RuntimeError("热词列表为空")

    regex = build_regex(hotwords)

    with Path(args.ref).open("r", encoding="utf-8") as ref_f, \
         Path(args.out).open("w", encoding="utf-8") as out_f:

        for line in ref_f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # 拆分出 ID 及句子内容
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            sentence = parts[1] if len(parts) > 1 else ""

            # 在整句字符串里直接做子串匹配
            hits = regex.findall(sentence)
            # 去重并保持原热词顺序
            seen = set()
            uniq_hits = [w for w in hotwords if w in hits and not (w in seen or seen.add(w))]

            # 写入 ocr.list
            if uniq_hits:
                out_f.write(utt_id + " " + " ".join(uniq_hits) + "\n")
            else:
                out_f.write(utt_id + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotwords", required=True, help="每行一个热词的 txt")
    parser.add_argument("--ref",      required=True, help="参考转写 token.ref")
    parser.add_argument("--out",      required=True, help="生成的 ocr.list")
    main(parser.parse_args())