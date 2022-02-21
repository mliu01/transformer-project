# -*- coding: utf-8 -*-
# %% [markdown]
# # MAKI4U Jumpstart Notebook
#
# A Notebook for training new BERT models for MAKI4U (former CCAI)\
# This is a refactored version of "bert_train_classifier.ipynb" from the
# BAS Jumpstart\ and is meant as optimization and general clean up of that notebook\
# It is possible to use this as notebook or directly as a script
#
#
# This notebook is organized in
# * [Configuration for Model and Logging](#config)
# * [Loading Dataset](#dataset)
# * [Model Definition](#model)
# * [Train Model](#train)
# %% [markdown]
# ## Imports
# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
import numpy as np
import math
import pandas as pd
from sklearn.metrics import classification_report
from IPython import get_ipython
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, DatasetDict, Dataset
import transformers
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding
)
from transformers.utils import check_min_version
import yaml
from utils.bert_custom_trainer import TrainerDiceLoss
from utils.configuration import (
    parse_arguments,
    save_config,
    yaml_dump_for_notebook,
    isnotebook,
)
from utils.metrics import compute_metrics
from utils.BERT import BERT
from typing import List, Set, Dict, Tuple, Optional

# %%
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.9.0.dev0")
transformers.logging.set_verbosity_info()

# %%
#get_ipython().run_line_magic("matplotlib", "inline")
# %% [markdown]
# ## Configuration and Logging <a class="anchor" id="config"></a>

# %%
# Extra variable for processing big files in BERT
# https://github.com/huggingface/datasets/issues/2181
block_size_10MB = 10 << 20

# %%
#if isnotebook():
    # In a notebook you need to just dump the yaml with the configuration details
args_dict = yaml_dump_for_notebook(filepath='configs/hierarchical-baseline.yml')
    # print(args_dict)
#else:
#    # This can only be used if this is run as a script. For notebooks use the yaml.dump and configure the yaml file accordingly
#    args, device = parse_arguments()
#    args_dict = args.__dict__

# %%
filename, filepath = save_config(args_dict)
print(filename, filepath)

# %%
writer = SummaryWriter(f"{filepath}/{filename}")


# %% [markdown]
# ## Loading Dataset <a class="anchor" id="dataset"></a>
# ### TODO: Refactor this properly
# %%
json_files = str(Path(args_dict["data_folder"]).joinpath(args_dict["data_file"]))
# %%
json_files_train = [json_files.replace(".json", "") + "_train.json"]
json_files_dev = [json_files.replace(".json", "") + "_dev.json"]
json_files_test = [json_files.replace(".json", "") + "_test.json"]

dataset_train = load_dataset(
    "json", data_files=json_files_train, chunksize=block_size_10MB
)["train"]

dataset_dev = load_dataset(
    "json", data_files=json_files_dev, chunksize=block_size_10MB
)["train"]

dataset_test = load_dataset(
    "json", data_files=json_files_test, chunksize=block_size_10MB
)["train"]

dataset_train = dataset_train.class_encode_column("label")
dataset_dev = dataset_dev.class_encode_column("label")
dataset_test = dataset_test.class_encode_column("label")

assert (
    dataset_train.features["label"].names == dataset_test.features["label"].names == dataset_dev.features["label"].names
), "Something went wrong, target_names of train and test should be the same"


# %%
tokenizer = AutoTokenizer.from_pretrained(
    args_dict["checkpoint_tokenizer"], use_fast=True, model_max_length=512
    )

# %%
dataset_train = dataset_train.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
    #num_proc=args_dict["tokenizer_num_processes"]
)

dataset_dev = dataset_dev.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
    #num_proc=args_dict["tokenizer_num_processes"]
)

dataset_test = dataset_test.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
    #num_proc=args_dict["tokenizer_num_processes"]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
#len(dataset_train[0]['input_ids'])

# %%
num_labels = dataset_train.features["label"].num_classes

# %%
# use shuffle to make sure the order of samples is randomized (deterministically)
dataset_train = dataset_train.shuffle(seed=args_dict["random_seed"])
dataset_dev = dataset_dev.shuffle(seed=args_dict["random_seed"])
#ds_train_testvalid = dataset_train.train_test_split(
#    test_size=(1 - args_dict["split_ratio_train"])
#)

dataset = DatasetDict(
    {
        "train": dataset_train,
        "valid": dataset_dev,
        "test": dataset_test
    }
)

target_names = dataset["test"].features["label"].names

if args_dict["oversampling"]:
    df_train = dataset["train"].to_pandas()
    min_samples = math.ceil(len(df_train) * args_dict["oversampling"])
    count_dict = dict(df_train["label"].value_counts())
    count_dict = {k: v for k, v in count_dict.items() if v < min_samples}

    over_samples = []
    for label_id, n_occurance in count_dict.items():
        class_samples = df_train[df_train["label"] == label_id]
        additional_samples = class_samples.sample(
            n=(min_samples - len(class_samples)), replace=True
        )
        over_samples.append(additional_samples)
        print(
            f"\nAdding {len(additional_samples)} samples for class {target_names[label_id]}"
        )

    new_train = pd.concat([df_train, *over_samples])
    dataset["train"] = Dataset.from_pandas(new_train)


dataset["train"] = dataset["train"].shuffle(seed=args_dict["random_seed"])

# %% [markdown]
# ## Model Definition <a class="anchor" id="model"></a>

# %%
# Model class definition was moved to utils for easier mentainance across notebooks
model_obj = BERT(
    args_dict, num_labels=num_labels
)

# %%
trainer_class = Trainer
if args_dict["custom_trainer"] == "TrainerDiceLoss":
    trainer_class = TrainerDiceLoss
    print("USING CUSTOM TRAINER CLASS")

# %%
training_args = TrainingArguments(
    filename,
    evaluation_strategy=args_dict["evaluation_strategy"],
    eval_steps=args_dict["evaluation_steps"],
    logging_dir=filepath,
    lr_scheduler_type=args_dict["lr_scheduler_type"],
    learning_rate=float(args_dict["lr_rate"]),
    warmup_ratio=args_dict["warm_up"],
    label_smoothing_factor=args_dict["label_smoothing"],
    per_device_train_batch_size=args_dict["batch_size"],
    per_device_eval_batch_size=args_dict["batch_size"],
    gradient_accumulation_steps=args_dict["gradient_accumulation_steps"],
    num_train_epochs=args_dict["epochs"],
    weight_decay=args_dict["weight_decay"],
    logging_strategy=args_dict["logging_strategy"],
    logging_steps=args_dict["logging_steps"],
    load_best_model_at_end=args_dict["load_best"],
    metric_for_best_model=args_dict["metric_used"],
    greater_is_better=args_dict["greater_better"],
    save_strategy=args_dict["save_strategy"],
    save_steps=args_dict["save_steps"],
    save_total_limit=args_dict["save_limits"],
    dataloader_num_workers=args_dict["workers"],
    disable_tqdm=False,
    remove_unused_columns=True,
    dataloader_drop_last=args_dict["load_best"]
)

# %% [markdown]
# ## Train Model <a class="anchor" id="Train"></a>
# %%
trainer = trainer_class(
    model_obj.model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=model_obj.tokenizer,
    compute_metrics=compute_metrics,
    data_collator = data_collator
)

trainer.train(resume_from_checkpoint=args_dict["resume_from_checkpoint"])

print("finished training")

# %%
eval_res = trainer.evaluate()
print(eval_res)

# %%
# Add to saving from here
Path("models").mkdir(parents=True, exist_ok=True)
trainer.save_model(f"models/{filename}")

# %%
logits, labels, metrics = trainer.predict(dataset["test"])
predictions = logits.argmax(1)

# %%
# TODO: Log this properly
log = classification_report(
        labels,
        predictions,
        #np.unique(labels),
        target_names=dataset["test"].features["label"].names,
    )
print(log)

# %%
print("done... saving model")

trainer.save_model(f"models/{filename}")
model_obj.model.save_pretrained(f"pretrained/{filename}")
model_obj.tokenizer.save_pretrained(f"pretrained/{filename}")

# %%
with open(f"pretrained/{filename}/classification_report.txt", 'w', encoding='utf-8') as f:
    f.write(log)

