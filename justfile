#!/usr/bin/env just -f

# Pathing
here := justfile_directory()
root := here
_just_run_prefix := "just --justfile " + root

# display help message
@help:
        just --list --unsorted
# Prep
prep:
    python ../nanoGPT/data/multiset/prepare.py
# Train
train:
    #    python train.py config/train_shakespeare_char.py
    python ../nanoGPT/train.py config/train_multiset.py
# Sample
sample:
    # python sample.py --out_dir=out-shakespeare-char
    python ../nanoGPT/sample.py --out_dir=out-multiset --device=cpu
