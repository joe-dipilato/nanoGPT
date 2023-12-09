# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = "out-multiset"
eval_interval = 150  # keep frequent because we'll overfit # This is how often we save
eval_iters = 100  # OR 200, 20. This one can probably start low, and slowly move higher.
eval_iters_min = 20
eval_iters_max = 200
log_interval = 5  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "multiset"
wandb_run_name = "mini-gpt"

dataset = "multiset"
gradient_accumulation_steps = 1
batch_size = 6  # OR 64, 12
batch_size_min = 6
batch_size_max = 64
block_size = 256  # context of up to 256 previous characters # OR 64

# baby GPT model :)
n_layer = 6  # 4, 6
n_head = 6  # 4, 6
n_embd = 384  # OR 384, 128
dropout = 0  # 0.2 -> 0

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 10000  # OR 2000
lr_decay_iters = 10000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = "cpu"  # run on cpu only mps
compile = True  # do not torch compile the model

init_from = "resume"


# import os
# import pickle

# data_dir = os.path.join("data", dataset)
# meta_path = os.path.join(data_dir, "meta.pkl")
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, "rb") as f:
#         meta = pickle.load(f)
