# -*- coding: utf-8 -*-
# %% [Markdown] [markdown]
# # Named Entity Recognition with BERT


# %% [markdown]
# This notebook is meant as a basepoint to perform custom NER tasks with BERT for MAKI4U (former CCAI)
#
# This notebook is organized in
# * [Configuration for Model and Logging](#config)
# * [Loading Dataset](#dataset)
# * [Model Definition](#model)

# %% [markdown]
# ## Imports

# %%
# For Testing Purpose to have a larger dataset
! wget http://noisy-text.github.io/2017/files/wnut17train.conll

# %%
#!pip install wandb -qqq
import wandb

wandb.login(relogin=False)

# %%
import sys

from IPython import get_ipython
import torch
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.utils import check_min_version
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional

# %%
from utils.bert_custom_trainer import TrainerDiceLoss
from utils.configuration import (
    parse_arguments,
    save_config,
    yaml_dump_for_notebook,
    isnotebook,
)
import utils.metrics
from utils.dataset import prep_conll_file, load_data
from utils.BERT import BERT
from utils.metrics import compute_metrics

# %%
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

# %% [markdown]
# ## Configuration and Logging <a class="anchor" id="config"></a>

# %%
# Extra variable for processing big files in BERT
# https://github.com/huggingface/datasets/issues/2181
block_size_10MB = 10 << 20

# %%
if isnotebook():
    # In a notebook you need to just dump the yaml with the configuration details
    args_dict = yaml_dump_for_notebook("configs/ner-baseline.yml")
    print(args_dict)
else:
    # This can only be used if this is run as a script. For notebooks use the yaml.dump and configure the yaml file accordingly
    args, device = parse_arguments()
    args_dict = args.__dict__

# %%
filename, filepath = save_config(args_dict)

# %%
filepath

# %%
# run = wandb.init(project=filename,
#           config=args_dict)

# %% [markdown]
# ## Loading Dataset <a class="anchor" id="dataset"></a>

# %%
#TEST: This is to test the trainer on a larger dataset
#TEST: After this jump directly to the train_test_split from this point 
import re
def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_wnut('wnut17train.conll')

# %%
#TEST Just to look at the dataset indices dont matter
print(texts[0][10:17], tags[0][10:17], sep='\n')

# %%
args_dict["data_folder"]

# %%
inputfile = Path(args_dict["data_folder"]).joinpath("conll-train.conll")
outputfile = Path(args_dict["data_folder"]).joinpath("conll-train-clean.conll")
prep_conll_file(inputfile=inputfile, outputfile=outputfile)

# %%
train_samples = load_data(outputfile)  # ATTENTION
val_samples = load_data(outputfile)  # ATTTENTION SAME FILE JUST FOR TEST PURPOSE
samples = train_samples + val_samples
schema = ["_"] + sorted({tag for sentence in samples for _, tag in sentence})

# %%
schema


# %%
# samples

# %%
def get_datastructure(samples: List[List[Tuple[str, str]]]):
    """
    splits the samples list into one for text and one for labels
    """
    # print(samples)
    texts, tags = list(), list()

    for sentence in samples:

        data = [list(t) for t in zip(*sentence)]
        # print(data)

        texts.append(data[0])
        tags.append(data[1])
        # data.append([list(t) for t in zip(*sentence)])  # consume the zip generator into a list of lists

    return texts, tags


texts, tags = get_datastructure(samples)  # [ [[tokens],[tags]], [[tokens],[tags]] ]

# %%
# Simple Test to get a slice of the loaded texts in the dataset. Change indices to see other parts of the text
print(texts[0][5:10], tags[0][5:10], sep="\n")

# %%
train_texts, val_texts, train_tags, val_tags = train_test_split(
    texts, tags, test_size=1 - args_dict["split_ratio_train"]
)  # Caution: Testsize 50% because the file used here was only 2 examples long

# %%
# Dictonaries for easy look-ups between words and tags
unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# %%
unique_tags

# %% [markdown]
# ## Model Definition <a class="anchor" id="model"></a>

# %%
model_obj = BERT(args_dict, num_labels=len(unique_tags))

# %%
model_obj.tokenizer

# %%
# Get the encodings from the tokenizers
train_encodings = model_obj.tokenizer(
    train_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)
val_encodings = model_obj.tokenizer(
    val_texts,
    is_split_into_words=True,
    return_offsets_mapping=True,
    padding=True,
    truncation=True,
)


# %%
def encode_tags(
    tags: List[List[str]], encodings: Dict[str, List[List[int]]]
) -> List[List[int]]:
    """
    Encodes our str labels to encoded representations
    """
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


# %%
train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)


# %%
class NER_Dataset(torch.utils.data.Dataset):
    """
    Make our Dataset a custom pytorch dataset for easier feed to the network
    """

    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> torch.tensor:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


# %%
train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = NER_Dataset(train_encodings, train_labels)
val_dataset = NER_Dataset(val_encodings, val_labels)

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
    dataloader_drop_last=args_dict["load_best"],
)

trainer = Trainer(
    model=model_obj.model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
)

# %%
trainer.train()  # Error occurs because the dataset is to tiny. Tested with bigger datasets and works fine
run.finish()

# %%
eval_res = trainer.evaluate()
print(eval_res)

# %%
trainer.save_model(f"models/{filename}")
model_obj.model.save_pretrained(f"pretrained/{filename}")
model_obj.tokenizer.save_pretrained(f"pretrained/{filename}")
