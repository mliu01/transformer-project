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
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from transformers.utils import check_min_version
from utils.bert_custom_trainer import TrainerDiceLoss
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
args_dict = yaml_dump_for_notebook(filepath='configs/rnn-baseline.yml')

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
if args_dict['task_type']== 'flat-classification':
    if args_dict['data_lvl']:
        dataset_full = dataset_full.remove_columns("label")
        dataset_full = dataset_full.rename_column(f"lvl{args_dict['data_lvl']}", "label")

    dataset_full = dataset_full.class_encode_column("label")
    num_labels = dataset_full.features["label"].num_classes
else:
    num_labels = len(list(set(dataset_full['label'])))

# removes unnecessary columns
rmv_col = [col for col in dataset_full.column_names if col not in ['label', 'text']]
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
    target_names = np.unique(dataset["test"]["label"])
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
            f"\nAdding {len(additional_samples)} samples for class {label_id}"
        )

    new_train = pd.concat([df_train, *over_samples])
    dataset["train"] = Dataset.from_pandas(new_train)

dataset["train"] = dataset["train"].shuffle(seed=args_dict["random_seed"])

# %%
assert(
    len(set(dataset['train']['label']))  == len(set(dataset['valid']['label'])) == len(set(dataset['test']['label']))
), "Repeat cell above, different amount of labels in the dataset splits!"

# %%
target_names = list(set(dataset['train']['label']))

# %% [markdown]
# ## Model Definition <a class="anchor" id="model"></a>

# %%
# Model class definition 
model_obj = BERT(
    args_dict, num_labels=num_labels, dataset=dataset
)

# %%
if not args_dict['task_type'] == 'flat-classification':
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
    print("USING FLAT METRICS")

elif args_dict['task_type'] == 'lcpn-hierarchical-classification':
    evaluator = scorer.HierarchicalScorer(args_dict['experiment_name'], model_obj.get_tree(), decoder)
    evaluator = evaluator.compute_metrics_transformers_lcpn
    print("USING LCPN SCORER")

elif args_dict['task_type'] == 'rnn-hierarchical-classification':
    evaluator = scorer.HierarchicalScorer(args_dict['experiment_name'])
    evaluator = evaluator.compute_metrics_transformers_rnn
    print("USING RNN SCORER")

elif args_dict['task_type'] == 'dhc-hierarchical-classification':
    evaluator = scorer.HierarchicalScorer(args_dict['experiment_name'])
    evaluator = evaluator.compute_metrics_transformers_dhc
    print("USING DHC SCORER")


# %%
trainer = trainer_class(
    model=model_obj.model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=evaluator,
    data_collator=data_collator
    #callbacks=[EarlyStoppingCallback(early_stopping_patience = 10)]
)

# %%
trainer.train(resume_from_checkpoint=args_dict['resume_from_checkpoint'])

# %%
# Add to saving from here
Path(f"{filename}/models").mkdir(parents=True, exist_ok=True)
Path(f"{filename}/results").mkdir(parents=True, exist_ok=True)

result_collector = ResultCollector(args_dict['data_file'], filename)  

# %%
for split in ['train', 'valid', 'test']:
    result_collector.results['{}+{}'.format(args_dict['experiment_name'], split)] \
        = trainer.evaluate(dataset[split])
result_collector.persist_results(time.time())

# %%
prediction, labels, metrics = trainer.predict(dataset["test"])

# %%
with open(f'{filename}/results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# %%
if args_dict['task_type'] == 'flat-classification' or args_dict['task_type'] == 'NER':
     predictions = prediction.argmax(1)
     
     report = classification_report(
        labels, 
        predictions,
        target_names= dataset['test'].features['label'].names, #target_names
        output_dict=True
     )
     print(report)
     pd.DataFrame(report).T.to_csv(f'{filename}/results/classification_report.csv', sep=';')

# %%
if args_dict['task_type'] == 'lcpn-hierarchical-classification':
    preds = prediction.argmax(-1)
    test_pred = pd.DataFrame({'label': [decoder[label]['value'] for label in dataset['test'].labels]})
    test_pred['prediction'] = [decoder[pred]['value'] for pred in preds]

    log = classification_report(test_pred['label'], test_pred['prediction'], output_dict = True)
    pd.DataFrame(log).T.to_csv(f'{filename}/results/classification_report.csv', sep=';')

# %%
# if args_dict['task_type'] == 'rnn-hierarchical-classification':
#     label_list = [decoder[tuple(l)]['value'] for l in labels]
#     pred_list = [decoder[tuple(pred.argmax(-1))]['value'] for pred in prediction]
#     log = classification_report(label_list, pred_list, output_dict = True)
#     pd.DataFrame(log).T.to_csv(f'{filename}/results/classification_report.csv', sep=';')

# %%
if args_dict['task_type'] == 'dhc-hierarchical-classification' or args_dict['task_type'] == 'rnn-hierarchical-classification':
     label_list = []
     predcition_list = []
     labels_per_lvl = np.array(labels).transpose().tolist()
     preds_per_lvl = [list(pred.argmax(-1)) for pred in prediction]

     if args_dict['task_type'] == 'rnn-hierarchical-classification':
          preds_per_lvl = np.array(preds_per_lvl).transpose().tolist()
          
     for i in range(3):
          label_list.append([decoder[i][label]['value'] for label in labels_per_lvl[i]])
          predcition_list.append([decoder[i][pred]['value'] for pred in preds_per_lvl[i]])

     
     lvl = 0
     for y_label, y_pred in zip(label_list, predcition_list):
          log = classification_report(y_label, y_pred, output_dict = True)

          pd.DataFrame(log).T.to_csv(f'{filename}/results/classification_report{lvl+1}.csv', sep=';')

          np.save(Path(f'{filename}/results/').joinpath(f"confusion_lvl{lvl+1}.npy"), confusion_matrix(y_label, y_pred))

          array = np.load(Path(f'{filename}/results').joinpath(f"confusion_lvl{lvl+1}.npy"))

          df_cm = pd.DataFrame(array, index = set(y_label), columns = set(y_label))
          plt.figure(figsize = (10,7))
          sn.heatmap(df_cm, annot=True)
          
          lvl += 1

# %%
print("done... saving model")

trainer.save_model(f"{filename}/models")
model_obj.model.save_pretrained(f"{filename}/pretrained")
tokenizer.save_pretrained(f"{filename}/pretrained")
