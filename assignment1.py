# %% [markdown]
# # CSE158 Assignment 1
# Run All to write `task_1_predictions.csv` … `task_6_predictions.csv`. Methods are described in `writeup.txt`; code is in `task1.py` … `task6.py` and `recsys_lib.py`. Restart the kernel after editing any `.py` file.
# 
# ### [Other Code Found in this Repo](https://github.com/balexdalex42/Assignment1_CodeFiles)
# 

# %%
%load_ext autoreload
%autoreload 2

import os
# fewer "out of memory" errors from fragmentation; must be set before torch is imported
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import gc
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')

import task1, task2, task5, task6


def free_gpu():
    """Release the GPU memory held by the previous task before the next one starts."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## Task 6: time to next interaction

# %%
bundle = task6.train("task_6_train.csv")
task6.predict(bundle, "task_6_test.csv", "task_6_predictions.csv")
del bundle
free_gpu()

# %% [markdown]
# ## Task 4: which item, with timestamp

# %%
# 100k held-out positives, each paired with 6 popularity-sampled negatives -> 600k combiner pairs
bundle = task2.train("tasks_3_4_train.csv", use_time=True, n_val=100000, n_neg=6)
task2.predict(bundle, "task_4_test.csv", "task_4_predictions.csv", use_time=True)
del bundle
free_gpu()

# %% [markdown]
# ## Task 2: which item

# %%
bundle = task2.train("tasks_1_2_train.csv", n_val=100000, n_neg=6)
task2.predict(bundle, "task_2_test.csv", "task_2_predictions.csv")
del bundle
free_gpu()

# %% [markdown]
# ## Task 1: rating

# %%
bundle = task1.train("tasks_1_2_train.csv")
task1.predict(bundle, "task_1_test.csv", "task_1_predictions.csv")
del bundle
free_gpu()

# %% [markdown]
# ## Task 3: rating, with timestamp

# %%
bundle = task1.train("tasks_3_4_train.csv", use_time=True)
task1.predict(bundle, "task_3_test.csv", "task_3_predictions.csv", use_time=True)
del bundle
free_gpu()

# %% [markdown]
# ## Task 5: next item

# %%
bundle = task5.train("task_5_train.csv")
task5.predict(bundle, "task_5_test.csv", "task_5_predictions.csv")
del bundle
free_gpu()

# %% [markdown]
# ## Check the prediction files

# %%
def check(k):
    with open(f"task_{k}_test.csv", newline="") as f:
        test = list(csv.reader(f))
    with open(f"task_{k}_predictions.csv", newline="") as f:
        pred = list(csv.reader(f))
    n_in = len(test[0]) - 1
    ok = (pred[0] == test[0] and len(pred) == len(test)
          and all(p[:n_in] == t[:n_in] and p[-1] != "" for p, t in zip(pred[1:], test[1:])))
    col = test[0][-1]
    if col == "rating":
        ok = ok and all(0 <= float(p[-1]) <= 5 for p in pred[1:])
    elif col == "which":
        i1, i2 = test[0].index("item1"), test[0].index("item2")
        ok = ok and all(p[-1] in (p[i1], p[i2]) for p in pred[1:])
    elif col == "delta":
        ok = ok and all(float(p[-1]) >= 0 for p in pred[1:])
    print(f"task {k}: {len(pred) - 1} rows, {'OK' if ok else 'PROBLEM'}")

for k in range(1, 7):
    check(k)


