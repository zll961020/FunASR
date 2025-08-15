#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把缺失的语言/情绪/事件/ITN 标签补成默认值。
用法:
  python tools/add_default_tags.py \
         --input  data/aishell/train/train.jsonl \
         --output data/aishell/train/train_tagged.jsonl
"""

import json, argparse, sys, io, os

DEFAULT_TAGS = {
    "text_language": "<|zh|>",
    "emo_target": "<|NEUTRAL|>",
    "event_target": "<|Speech|>",
    "with_or_wo_itn": "<|withitn|>",
}

def process(in_file, out_file):
    with io.open(in_file, "r", encoding="utf-8") as fin, \
         io.open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            for k, v in DEFAULT_TAGS.items():
                obj.setdefault(k, v)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 jsonl")
    parser.add_argument("--output", required=True, help="补全后的 jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process(args.input, args.output)