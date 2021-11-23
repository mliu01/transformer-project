from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import glob
import torch
from typing import List, Set, Dict, Tuple, Optional

# TODO:
# Train-Test-Split
# Random-Seed
#


def parse_arguments():

    p = argparse.ArgumentParser(
        description="Train a BERT(like) model"
        "Has Hyperparameter and Logging capabilities to fully customize"
    )

    p.add_argument(
        "--config", type=argparse.FileType(mode="r"), default="configs/baseline.yml"
    )

    # ------- Basic Settings ------- #

    p.add_argument(
        "--task_type",
        help="type of model training you are performing",
        type=str,
        choices=["Classification", "NER"],
        default="Classification",
    )

    p.add_argument(
        "--device",
        help="Device (cpu, cuda, ...) on which the code should be run."
        "Setting it to auto, the code will be run on the GPU if possible.",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    p.add_argument(
        "--experiment_name",
        help="name to append to the tensorboard logs directory",
        default=None,
    )

    p.add_argument(
        "--experiment_name_suffix",
        help="name to append to the tensorboard logs directory",
        default=None,
    )

    p.add_argument(
        "--data_folder",
        help="folder to get training_data from",
        type=str,
        default="../data",
    )

    p.add_argument(
        "--data_file",
        help="file to get training_data from",
        type=str,
        default="dataset_v007_postkorb_Widerruf.json",
    )

    p.add_argument(
        "--random_seed",
        help="the seed used for reproducing random generated things",
        type=int,
        default=42,
    )

    # -------- Model settings ------- #

    p.add_argument(
        "--architecture",
        help="The architecture used for training (BERT variants)",
        type=str,
        default="BERT",
        choices=["BERT", "roBERTa"],
    )

    p.add_argument(
        "--resume_from_checkpoint",
        help="should the training be resumend from a existing checkpoint of this model",
        type=bool,
        default=False,
    )

    p.add_argument(
        "--checkpoint_model",
        help="what is the checkpoint path",
        type=str,
        default="uklfr/gottbert-base",
    )

    p.add_argument(
        "--checkpoint_tokenizer",
        help="what is the checkpoint path",
        type=str,
        default="uklfr/gottbert-base",
    )

    p.add_argument("--tokenizer_num_processes", help="...", type=int, default=10)

    p.add_argument("--workers", help="", type=int, default=1)

    p.add_argument(
        "--load_best",
        help="Whether or not to load the best model found during training at the end of training.",
        type=bool,
        default=True,
    )

    p.add_argument(
        "--greater_better",
        help="Defines if a higher score equales a better score given the used metrics.",
        type=bool,
        default=True,
    )

    p.add_argument(
        "--metric_used",
        help="The metric used to evaluate the best model",
        type=str,
        default="matthews_correlation",
    )

    # ------- Hyperparameters ------- #

    p.add_argument(
        "--batch_size", help="batch size to be used in training", type=int, default=40
    )

    p.add_argument(
        "--lr_rate", help="learning rate used for training", type=float, default=1e-05
    )

    p.add_argument(
        "--weight_decay",
        help="amount of weight used in training",
        type=float,
        default=5e-1,
    )

    p.add_argument(
        "--warm_up", help="training steps for linear warumup", type=float, default=1 / 5
    )

    p.add_argument(
        "--lr_scheduler_type",
        help="the scheduler type for changing the learning rate",
        type=str,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        default="linear",
    )

    p.add_argument("--epochs", help="epochs for training", type=int, default=10)

    p.add_argument(
        "--oversampling", help="oversampling ratio used", type=float, default=0.3
    )

    p.add_argument(
        "--freeze_layer",
        help="amount of layers to be frozen in the pretrained model",
        type=int,
        default=6,
    )

    p.add_argument(
        "--label_smoothing",
        help="The label smoothing factor to use."
        """Zero means no label smoothing, otherwise the underlying onehot-encoded labels
                      are changed from 0s and 1s to label_smoothing_factor/num_labels
                      and 1 - label_smoothing_factor + label_smoothing_factor/num_labels respectively.',
                   """,
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--custom_trainer",
        help="Custom Trainer setting",
        type=str,
        choices=["TrainerDiceLoss", None],
        default=None,
    )

    p.add_argument(
        "--split_ratio_train",
        help="Split ratio between Training and Test Dataset",
        type=int,
        default=0.9,
    )

    # -------- Logging / Evaluation Settings -------- #

    p.add_argument(
        "--logging",
        help="should tensorboard / wandb log the current run (bool)",
        type=bool,
        default=True,
    )

    p.add_argument(
        "--logging_dir",
        help="path to store the log files in",
        type=str,
        default=f"logs/",
    )

    p.add_argument(
        "--logging_strategy",
        help="what strategy to adopt during training",
        type=str,
        default="steps",
        choices=["steps", "epochs"],
    )

    p.add_argument(
        "--logging_steps", help="Logging is done every n_steps", type=int, default=50
    )

    p.add_argument(
        "--evaluation_strategy",
        help="what strategy to adopt during training",
        type=str,
        default="steps",
        choices=["steps", "epochs"],
    )

    p.add_argument(
        "--evaluation_steps",
        help="Evaluation is done every n_steps",
        type=int,
        default=50,
    )

    p.add_argument(
        "--save_strategy",
        help="what strategy to adopt during training",
        type=str,
        default="steps",
        choices=["steps", "epochs"],
    )

    p.add_argument(
        "--save_steps", help="Saving is done every n_steps", type=int, default=1000
    )

    p.add_argument(
        "--save_limit",
        help="Limits the amount of checkpoints and deletes the oldest one when limit would otherwise succeeded",
        type=int,
        default=5,
    )

    args = p.parse_args()

    print(args)

    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():

            if isinstance(value, list) and arg_dict[key] is not None:
                for v in value:
                    arg_dict[key].append(v)

            else:
                arg_dict[key] = value

    # args.learning_rate = float(args.learning_rate)

    if args.device == "cuda" or "auto" and torch.cuda.is_available():

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

    else:
        device = torch.device("cpu")
        print("USING CPU BECAUSE GPU IS NOT AVAILABLE")

    return args, device


def save_config(config: dict) -> Tuple[str, str]:
    """
    creates directories and config-yml for the current experiment settings
    returns: filename and filepath for logging
    """

    p = Path("log/")
    # print(config)
    p = p.joinpath(f"{config['task_type']}")
    p = p.joinpath(f"{config['architecture']}")
    p.mkdir(parents=True, exist_ok=True)

    filename = file_naming(p, config)
    filepath = p / filename

    with open(f"{str(filepath)}.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, indent=4)

    return filename, filepath


def file_naming(path: Path, config: dict) -> str:
    """versions the file and names it according to attributes or the defined name"""

    # Create a string either based on parameters or the explicit experiment name and append a "_V" for Version
    if (not config["experiment_name"]) or (config["experiment_name"] == "None"):
        filename = (
            f"E{config['epochs']}_B{config['batch_size']}_LR{config['lr_rate']}_V"
        )
    else:
        filename = f"{config['experiment_name']}{config['experiment_name_suffix']}_V"

    # Search all already existing files with the given name
    file_list = [name for name in glob.glob(f"{path}/{filename}[0-9][0-9][0-9].yml")]

    # If there are already files with that name
    if file_list:

        # sort the list to get the newest version e.g. [file_V001.yml, file_V002.yml, file_V003.yml] pops "file_V003.yml"
        # [-6:-4] subscribes the Version number file_V**003**.yml --> 003. When converted to int we get 3 and can increment that version by 1
        i = int(sorted(file_list).pop()[-6:-4]) + 1
        i = "00" + str(i)
        i = i[-3:]
        filename += i

    else:
        filename += "001"

    # This filename does not include the file ending to generally use it for all files corresponding to this experiment
    return f"{filename}"


def yaml_dump_for_notebook(filepath="configs/baseline.yml"):
    # Call this function if you use this as a notebook so bypass argparse and directly dump the config
    with open(str(filepath), "r", encoding="utf-8") as stream:
        args_dict = yaml.safe_load(stream) or dict()

    return args_dict


def isnotebook():
    """
    a simple function to quickly check if something is running in a notebook
    Important for example for the use of argparse
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
