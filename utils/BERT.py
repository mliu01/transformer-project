import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    set_seed,
)


class BERT(nn.Module):
    def __init__(
        self, args_dict: dict, device: str = "cuda", num_labels: int = 10,
    ):
        """Loads the model and freezes the layers

        Args:
            args_dict (dict): argparse dictonary
            model_name (str, optional): hugging face model to be used. Defaults to "uklfr/gottbert-base".
            tokenizer_checkpoint: same as above for tokenizer
            num_labels (int, optional): number of labels. 1 for Regression. Default to 10 classes.
        """
        super(BERT, self).__init__()

        self.config, self.unused_kwargs = AutoConfig.from_pretrained(
            args_dict["checkpoint_model"],
            output_attentions=True,
            num_labels=num_labels,
            return_dict=True,
            return_unused_kwargs=True,
        )

        if args_dict["task_type"] == "classification":
            self.model = AutoModelForSequenceClassification.from_config(self.config)
        elif args_dict["task_type"] == "NER":
            self.model = AutoModelForTokenClassification.from_config(self.config)
        else:
            raise Exception(
                "Task unknown. Add the new task or use a kown model ('classification', 'NER')"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            args_dict["checkpoint_tokenizer"], use_fast=True
        )

        print(self.model)

        if args_dict["freeze_layer"]:
            self.freeze(args_dict["freeze_layer"], model_arc=args_dict["architecture"])

    def freeze(self, n_freeze_layer: int, model_arc: str = "roberta"):
        """freezes the layer of the bert model and adds a unfrozen classifier at the end
        """
        # TODO: work in model.named_modules() to get all layers and make this code architecture independet
        # TODO: Through that we can scrap the model_arc argument in the parser entirly and use every bert model architecture out there
        # Catch the difference between RoBERTa and BERT model architecture
        if model_arc == "roberta":
            model_arc = self.model.roberta

        elif model_arc == "bert":
            model_arc = self.model.bert

        elif model_arc == "distilbert":
            model_arc = self.model.distilbert

        else:
            raise Exception(
                "Architecture unknown. Add the new architecture or use a kown model ('bert', 'roberta', 'distilbert')"
            )

        if model_arc == "roberta" or model_arc == "bert":
            max_bert_layer = len(model_arc.encoder.layer)
            freezable_modules = [
                model_arc.encoder.layer,
                *model_arc.encoder.layer[: min(n_freeze_layer, max_bert_layer)],
            ]
        else:
            max_bert_layer = len(model_arc.transformer.layer)
            freezable_modules = [
                model_arc.transformer.layer,
                *model_arc.transformer.layer[: min(n_freeze_layer, max_bert_layer)],
            ]

        # Default RoBERTa model "gottBERT" has 12 layers
        # Default DistilBERT has 6 layers
        for module in freezable_modules:
            for param in module.parameters():
                param.requires_grad = False

        for k, v in model_arc.named_parameters():
            if v.requires_grad:
                print("{}: {}".format(k, v.requires_grad))
