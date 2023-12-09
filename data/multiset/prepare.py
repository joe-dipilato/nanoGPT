"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle

import numpy as np
import requests

"""
https://www.gutenberg.org/browse/loccs/ae
https://www.gutenberg.org/cache/epub/56796/pg56796.txt
https://www.gutenberg.org/cache/epub/200/pg200.txt
https://www.gutenberg.org/cache/epub/13600/pg13600.txt
https://www.gutenberg.org/cache/epub/34018/pg34018.txt
2
"""
# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url1 = [14006, 19719, 15097, 14070, 7188, 5430, 7010, 754]
    full_url1 = [
        f"https://www.gutenberg.org/ebooks/{urlid}.txt.utf-8" for urlid in data_url1
    ]
    data_url2 = [
        200,
        11615,
        13600,
        19699,
        19846,
        27478,
        27479,
        27480,
        30073,
        30685,
        30935,
        30976,
        31156,
        31329,
        31447,
        31641,
        31793,
        31855,
        31950,
        32063,
        32097,
        32182,
        32294,
        32423,
        32607,
        32689,
        32758,
        32783,
        32860,
        32940,
        32975,
        33052,
        33127,
        33189,
        33239,
        33295,
        33365,
        33427,
        33477,
        33550,
        33614,
        33698,
        33750,
        33991,
        34018,
        34047,
        34073,
        34074,
        34075,
        34082,
        34116,
        34162,
        34209,
        34312,
        34405,
        34533,
        34612,
        34702,
        34751,
        34878,
        34992,
        35092,
        35169,
        35236,
        35306,
        35398,
        35473,
        35561,
        35606,
        35747,
        35843,
        35844,
        35845,
        35925,
        36104,
        36226,
        36452,
        36735,
        37064,
        37160,
        37282,
        37461,
        37523,
        37610,
        37736,
        37806,
        37880,
        37984,
        38143,
        38202,
        38304,
        38401,
        38454,
        38539,
        38622,
        38709,
        38799,
        38892,
        38964,
        39029,
        39127,
        39232,
        39353,
        39435,
        39521,
        39632,
        39700,
        39775,
        39908,
        40009,
        40096,
        40156,
        40370,
        40538,
        40641,
        40769,
        40863,
        40956,
        41055,
        41156,
        41264,
        41343,
        41472,
        41567,
        41685,
        41773,
        41902,
        42048,
        42173,
        42342,
        42473,
        42552,
        42638,
        42736,
        42854,
        43060,
        43254,
        43427,
        56796,
    ]
    full_url2 = [
        f"https://www.gutenberg.org/cache/epub/{urlid}/pg{urlid}.txt"
        for urlid in data_url2
    ]
    full_url = full_url1 + full_url2
    with open(input_file_path, "w") as f:
        for url in full_url:
            print(url)
            f.write(requests.get(url, timeout=8).text)

with open(input_file_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


# create the train and test splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

from collections import Counter

data_counter = Counter()
letters = "abcdefghijklmnopqrstuvwxyz"
for idx, _ in enumerate(val_data[:100000]):
    for count in [0]:
        val = val_data[idx : idx + count + 2]
        if val[-1] not in letters or val[0] not in letters:
            break
        data_counter[val] += count + 2
max_tokens = 256
extra_num = 256 - len(chars)
# superset_tokens = data_counter.most_common(extra_num * 2)
# for idx, _ in enumerate(superset_tokens):
#     tok, val = superset_tokens[idx]
#     for itok, ival in superset_tokens[:idx]:
#         if itok in tok and ival > val / 2:
#             print(tok)
#             print(itok)

extra_tokens = data_counter.most_common(extra_num)
all_tokens = [l for l in chars] + [letter for letter, _ in extra_tokens]

stoi = {v: i for i, v in enumerate(all_tokens)}
itos = {i: v for i, v in enumerate(all_tokens)}

print(all_tokens)
vocab_size = len(all_tokens)


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
# val has 111540 tokens
# val has 111540 tokens
