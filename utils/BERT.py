import torch.nn as nn
from transformers import (
    AutoConfig,
    set_seed,
)
from utils.category_dataset import CategoryDatasetHierarchy
from utils.model_runner import provide_model_and_tokenizer
from utils.tree_utils import TreeUtils

class BERT(nn.Module):
    def __init__(
        self, args_dict: dict, device: str = "cuda", num_labels: int = 10, num_labels_per_lvl=None
    ):
        """Loads the model and freezes the layers

        Args:
            args_dict (dict): argparse dictonary
            model_name (str, optional): hugging face model to be used. Defaults to "uklfr/gottbert-base".
            tokenizer_checkpoint: same as above for tokenizer
            num_labels (int, optional): number of labels. 1 for Regression. Default to 10 classes.
        """
        super().__init__()

        self.config, self.unused_kwargs = AutoConfig.from_pretrained(
            args_dict["checkpoint_model_or_path"],
            hidden_dropout_prob=args_dict['hidden_dropout_prob'],
            output_attentions=False,
            num_labels=num_labels,
            return_dict=True,
            return_unused_kwargs=True,
        )

        if args_dict["task_type"] and args_dict["checkpoint_model_or_path"]:
            if args_dict["task_type"] == "hierarchical-classification" or args_dict["task_type"] == "rnn-hierarchical-classification":
                self.config.num_labels = num_labels
                self.config.num_labels_per_lvl = num_labels_per_lvl

            elif args_dict["task_type"] == "lcpn-hierarchical-classification":
                raise Exception(
                    "LCPN-Hierarchical-Classification doesn't work properly for now. Needs debug."
                    )

            self.tokenizer, self.model = provide_model_and_tokenizer(args_dict["task_type"], args_dict["checkpoint_model_or_path"], self.config)
        else:
            raise Exception(
                "Task unknown. Add the new task or use a kown model ('classification', 'NER')"
            )

        self.model.to(args_dict['device'])

        if args_dict["freeze_layer"]:
            self.freeze(args_dict["freeze_layer"], model_typ=args_dict["architecture"], task_type=args_dict["task_type"])


    def freeze(self, n_freeze_layer: int, model_typ: str = "roberta", task_type="flat-classification"):
        """freezes the layer of the bert model and adds a unfrozen classifier at the end
        """
        # TODO: work in model.named_modules() to get all layers and make this code architecture independet

        # This if statement catches the different namings of layers in the model architecture 
        model_arc_full = False
        if model_typ == "roberta":
            if task_type == "hierarchical-classification":
                model_arc = self.model.model
                model_arc_full = self.model
            else:
                model_arc = self.model.roberta

        elif model_typ == "bert":
            if task_type == "hierarchical-classification":
                model_arc = self.model.model
                model_arc_full = self.model
            else:
                model_arc = self.model.bert

        elif model_typ == "distilbert":
            model_arc = self.model.distilbert

        else:
            raise Exception(
                "Architecture unknown. Add the new architecture or use a kown model ('bert', 'roberta', 'distilbert')"
            )

        if model_typ == "roberta" or model_typ == "bert":
            max_bert_layer = len(model_arc.encoder.layer)
            freezable_modules = [
                model_arc.embeddings,
                *model_arc.encoder.layer[: min(n_freeze_layer, max_bert_layer)], # Defines which layers are to be frozen according to the chosen hyperparameters
            ]
        else:  # Must be DistilBert, all unkown architectures have been filtered in the step before. In case a new architecure is added (apart from distil, roberta and bert) modify the if statement accordingly
            max_bert_layer = len(model_arc.transformer.layer)
            freezable_modules = [
                model_arc.embeddings,
                *model_arc.transformer.layer[: min(n_freeze_layer, max_bert_layer)], # Defines which layers are to be frozen according to the chosen hyperparameters
            ]

        # Default RoBERTa model "gottBERT" has 12 layers
        # Default DistilBERT has 6 layers
        for module in freezable_modules:
            for param in module.parameters():
                param.requires_grad = False # This means we do not want to change the weights of the layer throughout the training process

        if model_arc_full:
            for k, v in model_arc_full.named_parameters():
                if v.requires_grad:
                    print("{}: {}".format(k, v.requires_grad)) # Prints a list of all layers that can still be trained after freezing
        else:
            for k, v in model_arc.named_parameters():
                if v.requires_grad:
                    print("{}: {}".format(k, v.requires_grad)) # Prints a list of all layers that can still be trained after freezing