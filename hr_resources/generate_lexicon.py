#!/usr/bin/env python3

import jieba
from pypinyin import Style, lazy_pinyin, load_phrases_dict, phrases_dict, pinyin_dict

user_defined = {
    "行长": [["hang2"], ["zhang3"]],
    "银行行长": [["yin2"], ["hang2"], ["hang2"], ["zhang3"]],
}


load_phrases_dict(user_defined)


def main():
    filename = "lexicon.txt"

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict
    phrases.update(**user_defined)

    i = 0
    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)
            tokens = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]
            if tokens[-1] not in "01234":
                tokens += "1"

            if len(tokens) < 3:
                tokens = tokens[0] + tokens

            f.write(f"{w} {tokens}\n")

        for key in phrases:
            tokens = lazy_pinyin(key, style=Style.TONE3, tone_sandhi=True)
            for i, t in enumerate(tokens):
                if t[-1] not in "01234":
                    tokens[i] += "1"

                if len(tokens[i]) < 3:
                    tokens[i] = tokens[i][0] + tokens[i]

            tokens = " ".join(tokens)

            f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
