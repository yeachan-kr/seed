import os
import random
import torch
import numpy as np
from pathlib import Path
from tokenizers.normalizers import BertNormalizer

f = open('util/vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
mappings = {m[0]: m[2:] for m in mappings}
f.close()

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def init_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # one gpu
    torch.cuda.manual_seed_all(seed)    # multi gpu
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n* Set a random seed to {seed}\n")


def float_separator(num: int) -> str:
    num_seg = []
    while num > 1000:
        num_seg.append(num % 1000)
        num = num // 1000

    str_num = [num] + num_seg[::-1]
    temp = []
    for i, n in enumerate(str_num):
        if n == 0:
            temp.append('000')
        elif (i != 0) and (n < 100):
            temp.append('0' + str(n))
        else:
            temp.append(str(n))
    str_num = ','.join(temp)
    return str_num


def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)
