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
#%load_ext autoreload
#%autoreload 2

# %%
import time
import csv

from pathlib import Path
import json
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn

from IPython import get_ipython
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
from utils.bert_custom_trainer import TrainerLossNetwork, TrainerDiceLoss
from utils.configuration import (
    parse_arguments,
    save_config,
    yaml_dump_for_notebook,
    isnotebook
)
from utils.metrics import Metrics
from utils import scorer
from utils.BERT import BERT
from utils.result_collector import ResultCollector
from utils.custom_dataset_encoding import CustomDS

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
args_dict = yaml_dump_for_notebook(filepath='configs/hierarchical-baseline.yml')

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
dataset_full = load_dataset(
    "json", data_files=json_files, chunksize=block_size_10MB
)["train"]

# %%
if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'NER':
    if args_dict['data_lvl']:
        dataset_full = dataset_full.remove_columns("label")
        dataset_full = dataset_full.rename_column(f"lvl{args_dict['data_lvl']}", "label")

    dataset_full = dataset_full.class_encode_column("label")

elif args_dict['task_type'] == 'hierarchical-classification':
    if 'path_list' in dataset_full.column_names:

        #encoding
        dataset_full = dataset_full.rename_column("label", "leaf_label")

        ds_object = CustomDS(dataset_full)
        dataset_full = ds_object.get_encoded_dataset()


# removes unnecessary columns
rmv_col = [col for col in dataset_full.column_names if col not in ['label', 'leaf_label', 'text']]
dataset_full = dataset_full.remove_columns(rmv_col)
dataset_full

# %%
tokenizer = AutoTokenizer.from_pretrained(
    args_dict["checkpoint_model_or_path"], use_fast=True, model_max_length=512
    )

# %%
dataset_full = dataset_full.map(
    lambda x: tokenizer(x['text'], truncation=True), 
    batched=True
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# use shuffle to make sure the order of samples is randomized (deterministically)
dataset_full = dataset_full.shuffle(seed=args_dict["random_seed"])
ds_train_testvalid = dataset_full.train_test_split(test_size=(1 - args_dict["split_ratio_train"])) #split full to train - test
test_valid = ds_train_testvalid['test'].train_test_split(test_size=0.5) #split test to validaton - test (1:1)


dataset = DatasetDict(
    {
        "train": ds_train_testvalid['train'],
        "valid": test_valid['train'],
        "test": test_valid['test']
    }
)

if args_dict["oversampling"]:
    target_names = np.unique(dataset["test"]["leaf_label"])
    df_train = dataset["train"].to_pandas()
    min_samples = math.ceil(len(df_train) * args_dict["oversampling"])
    count_dict = dict(df_train["leaf_label"].value_counts())
    count_dict = {k: v for k, v in count_dict.items() if v < min_samples}

    over_samples = []
    for label_id, n_occurance in count_dict.items():
        class_samples = df_train[df_train["leaf_label"] == label_id]
        additional_samples = class_samples.sample(
            n=(min_samples - len(class_samples)), replace=True
        )
        over_samples.append(additional_samples)
        print(
            f"\nAdding {len(additional_samples)} samples for class {label_id}"
        )

    new_train = pd.concat([df_train, *over_samples])
    dataset["train"] = Dataset.from_pandas(new_train)

dataset["train"] = dataset["train"].shuffle(seed=args_dict["random_seed"])

# %%
from utils.custom_dataset_encoding import load_encoder
encoder = load_encoder()
num_labels = len(encoder[2].classes_)
num_lables_per_lvl = {i: len(encoder[i].classes_) for i in range(len(encoder))}

# %%
num_lables_per_lvl

# %% [markdown]
# ## Model Definition <a class="anchor" id="model"></a>

# %%
# Model class definition 
model_obj = BERT(
    args_dict, num_labels=num_labels, num_labels_per_lvl=num_lables_per_lvl
)

# %%
trainer_class = Trainer
if args_dict["custom_trainer"] == "TrainerLossNetwork":
   trainer_class = TrainerLossNetwork
   print("USING CUSTOM TRAINER: TrainerLossNetwork")
if args_dict["custom_trainer"] == "TrainerDiceLoss":
   trainer_class = TrainerDiceLoss
   print("USING CUSTOM TRAINER: TrainerDiceLoss")

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

# %% [markdown]
# ## Train Model <a class="anchor" id="Train"></a>

# %%
if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'NER':
    evaluator = Metrics(dataset_full.features['label'].names).compute_metrics

elif args_dict['task_type'] == 'hierarchical-classification':
    evaluator = scorer.HierarchicalScorer(args_dict['experiment_name']).compute_metrics

    assert (
        all(len(elem)==3 for elem in dataset['train']['label'])
    ), "Something went wrong during encoding, all labels should have length of 3 (ignore if hierarchy level is not 3)"

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
#dataset['train']['label']

# %%
trainer.train(resume_from_checkpoint=args_dict['resume_from_checkpoint'])

# %%
# Add to saving from here
Path(f"{filename}/models").mkdir(parents=True, exist_ok=True)
result_collector = ResultCollector(args_dict['data_file'], filename)  

# %%
#result_collector = ResultCollector(args_dict['data_file'], filename)  
for split in ['train', 'valid', 'test']:
    result_collector.results['{}+{}'.format(args_dict['experiment_name'], split)] \
        = trainer.evaluate(dataset[split])
result_collector.persist_results(time.time())

# %%
if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'NER':
     logits, labels, metrics = trainer.predict(dataset["test"])
     predictions = logits.argmax(1)
     
     report = classification_report(
        labels, 
        predictions,
        target_names= dataset['test'].features['label'].names #target_names
     )
     print(report)     
     with open(f'{filename}/results/metrics.json', 'w') as metrics:
          json.dump(report, metrics, indent=4)

# %%
prediction = trainer.predict(dataset['test'])

# %%
#if args_dict['task_type'] == 'hierarchical-classification':
preds =  np.array(np.array([list(pred.argmax(-1)) for pred in prediction.predictions]).transpose().tolist())

label_list = []
predcition_list = []
for i in range(3):
     label_list.append([encoder[i].inverse_transform([label]) for label in prediction.label_ids[:, i]])
     predcition_list.append([encoder[i].inverse_transform([prediction]) for prediction in preds[:, i]])

test_pred=pd.DataFrame(data={
"label_lvl1": label_list[0] ,"prediction_lvl1": predcition_list[0], 
"label_lvl2": label_list[1] ,"prediction_lvl2": predcition_list[1],
"label_lvl3": label_list[2] ,"prediction_lvl3": predcition_list[2]
}) 

assert (
len(dataset['test']) == len(test_pred)
), "Something went wrong, length of test datasets should be the same"

full_prediction_output = '{}/{}.csv'.format(f"{filename}/results", "prediction-results")
test_pred.to_csv(full_prediction_output, index=False, sep=';', encoding='utf-8', quotechar='"',
     quoting=csv.QUOTE_ALL)

# %%
lvl = 1
for label, prediction in zip(label_list, predcition_list):
    np.save(Path(f'{filename}/results/').joinpath(f"confusion_lvl{lvl}.npy"), confusion_matrix(label, prediction))
    lvl += 1

# %%
# results hierarchy level 1
log1 = classification_report(
        label_list[0], 
        predcition_list[0],
        output_dict = True
    )
#print(log1)
with open(f'{filename}/results/classification_report1.json', 'w') as metrics:
    json.dump(log1, metrics, indent=4)

label = set(label_list[0])
array = np.load(Path(f'{filename}/results').joinpath("confusion_lvl1.npy"))

df_cm = pd.DataFrame(array, index = label,
                  columns = label)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

# %%
# results hierarchy level 2
log2 = classification_report(
        label_list[1], 
        predcition_list[1],
        output_dict = True
    )
#print(log2)
with open(f'{filename}/results/classification_report2.json', 'w') as metrics:
    json.dump(log2, metrics, indent=4)

label = set(label_list[1])
array = np.load(Path(f'{filename}/results/').joinpath("confusion_lvl2.npy"))

df_cm = pd.DataFrame(array, index = label,
                  columns = label)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

# %%
# results hierarchy level 3
log3 = classification_report(
        label_list[2], 
        predcition_list[2],
        output_dict = True
    )
#print(log3)
with open(f'{filename}/results/classification_report3.json', 'w') as metrics:
    json.dump(log3, metrics, indent=4)

label = set(label_list[2])
array = np.load(Path(f'{filename}/results/').joinpath("confusion_lvl3.npy"))

df_cm = pd.DataFrame(array, index = label,
                  columns = label)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

# %%
print("done... saving model")

trainer.save_model(f"{filename}/models")
model_obj.model.save_pretrained(f"{filename}/pretrained")
model_obj.tokenizer.save_pretrained(f"{filename}/pretrained")

# %%
#TODO: CLEAN UP CODE