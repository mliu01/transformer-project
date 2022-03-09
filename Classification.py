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
%load_ext autoreload
%autoreload 2

# %%
import time
import csv

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
    DataCollatorWithPadding,
    EarlyStoppingCallback
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
from utils import scorer
from utils.BERT import BERT
from utils.result_collector import ResultCollector

# %%
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.9.0.dev0")
transformers.logging.set_verbosity_info()

# %%
get_ipython().run_line_magic("matplotlib", "inline")

# %% [markdown]
# ## Configuration and Logging <a class="anchor" id="config"></a>

# %%
# Extra variable for processing big files in BERT
# https://github.com/huggingface/datasets/issues/2181
block_size_10MB = 10 << 20

# %%
#if isnotebook():
    # In a notebook you need to just dump the yaml with the configuration details
args_dict = yaml_dump_for_notebook(filepath='configs/flat-baseline.yml')
#else:
#    # This can only be used if this is run as a script. For notebooks use the yaml.dump and configure the yaml file accordingly
#    args, device = parse_arguments()
#    args_dict = args.__dict__

# %%
filename, filepath = save_config(args_dict)
print(filename, filepath)

# %%
writer = SummaryWriter(f"{filepath}")

# %% [markdown]
# ## Loading Dataset <a class="anchor" id="dataset"></a>
# ### TODO: Refactor this properly

# %%
json_files = str(Path(args_dict["data_folder"]).joinpath(args_dict["data_file"]))

# %%
json_files_train = [json_files.replace(".json", "") + "_train.json"]
json_files_test = [json_files.replace(".json", "") + "_test.json"]

dataset_train = load_dataset(
    "json", data_files=json_files_train, chunksize=block_size_10MB
)["train"]

dataset_test = load_dataset(
    "json", data_files=json_files_test, chunksize=block_size_10MB
)["train"]

if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'classification':
    dataset_train = dataset_train.class_encode_column("label")
    dataset_test = dataset_test.class_encode_column("label")

assert (
    set(dataset_train['label']) == set(dataset_test['label'])
), "Something went wrong, target_names of train and test should be the same"

# %%
tokenizer = AutoTokenizer.from_pretrained(
    args_dict["checkpoint_tokenizer"], use_fast=True, model_max_length=512
    )

# %%
dataset_train = dataset_train.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
)

dataset_test = dataset_test.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
num_labels = len(np.unique(dataset_train['label']))

# %%
# use shuffle to make sure the order of samples is randomized (deterministically)
dataset_train = dataset_train.shuffle(seed=args_dict["random_seed"])
ds_train_testvalid = dataset_train.train_test_split(
    test_size=(1 - args_dict["split_ratio_train"])
)

dataset = DatasetDict(
    {
        "train": ds_train_testvalid['train'],
        "valid": ds_train_testvalid['test'],
        "test": dataset_test
    }
)

target_names = np.unique(dataset["test"]["label"])

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
    args_dict, num_labels=num_labels, dataset = dataset
)

# %%
decoder = model_obj.get_decoder()
train_set, dev_set, test_set = model_obj.get_datasets()
dataset = DatasetDict(
{
    "train": train_set,
    "valid": dev_set,
    "test": test_set
}
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
    dataloader_drop_last=args_dict["drop_last"]
)

# %%
if args_dict['checkpoint_torch_model']:
    checkpoint = torch.load(args_dict['checkpoint_torch_model'], map_location='cuda')
    model_obj.model.load_state_dict(checkpoint['model_state_dict'])

# %% [markdown]
# ## Train Model <a class="anchor" id="Train"></a>

# %%
if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'classification':
    evaluator = compute_metrics
    
elif args_dict['task_type'] == 'hierarchical-classification':
    evaluator = scorer.HierarchicalScorer(args_dict['experiment_name'], model_obj.get_tree(), decoder)
    evaluator = evaluator.compute_metrics_transformers_flat

# %%
trainer = trainer_class(
    model_obj.model,
    training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=model_obj.tokenizer,
    compute_metrics=evaluator,
    data_collator=data_collator,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience = 10)]
)

# %%
trainer.train(resume_from_checkpoint=args_dict["resume_from_checkpoint"])

print("finished training")

# %%
# Add to saving from here
Path(f"{filename}/models").mkdir(parents=True, exist_ok=True)
Path(f"{filename}/torch_pretrained").mkdir(parents=True, exist_ok=True)

# %%
result_collector = ResultCollector(args_dict['data_file'], filename)  
for split in ['train', 'valid', 'test']:
    result_collector.results['{}+{}'.format(args_dict['experiment_name'], split)] \
        = trainer.evaluate(dataset[split])

# %%
prediction = trainer.predict(dataset['test'])
preds = prediction.predictions.argmax(-1)

test_pred = pd.DataFrame({'label': dataset_test['label']})
assert (
     len(dataset['test']) == len(test_pred)
), "Something went wrong, length of test datasets should be the same"

if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'classification':
    p = [pred for pred in preds]
    
elif args_dict['task_type'] == 'hierarchical-classification':
     p = [decoder[pred]['value'] for pred in preds]
     
test_pred['predicition'] = p

# %%
# Persist results
result_collector.persist_results(time.time())

full_prediction_output = '{}/{}.csv'.format(f"{filename}/results", "prediction-results")
test_pred.to_csv(full_prediction_output, index=False, sep=';', encoding='utf-8', quotechar='"',
                quoting=csv.QUOTE_ALL)

# %%
# TODO: Log this properly, inaccurate
# log = classification_report(
#         prediction.label_ids, 
#         preds,
#         target_names= target_names
#     )
# print(log)

# with open(f"{filename}/results/classification_report.txt", 'w', encoding='utf-8') as f:
#     f.write(log)

# %%
print("done... saving model")

trainer.save_model(f"{filename}/models")
model_obj.model.save_pretrained(f"{filename}/pretrained")
model_obj.tokenizer.save_pretrained(f"{filename}/pretrained")
model_obj.model.save(model_obj.model, trainer.optimizer, f"{filename}/torch_pretrained/model.pth")
