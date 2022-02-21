import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoConfig,
    set_seed,
)
from HierarchicalClassificationHead import ClassificationModel
from utils.tree_utils import TreeUtils
from utils.category_dataset_flat import CategoryDatasetFlat
from pathlib import Path
import pickle

import pandas as pd

class BERT(nn.Module):
    def __init__(
        self, args_dict: dict, device: str = "cuda", num_labels: int = 10, dataset=None
    ):
        """Loads the model and freezes the layers

        Args:
            args_dict (dict): argparse dictonary
            model_name (str, optional): hugging face model to be used. Defaults to "uklfr/gottbert-base".
            tokenizer_checkpoint: same as above for tokenizer
            num_labels (int, optional): number of labels. 1 for Regression. Default to 10 classes.
        """
        super(BERT, self).__init__()
        self.dataset = dataset
        self.tree = None
        self.root = None
        
        self.data_name = args_dict['data_file'].split(".")[0]
        self.load_tree()      

        self.config, self.unused_kwargs = AutoConfig.from_pretrained(
            args_dict['checkpoint_model'],
            output_attentions=False,
            num_labels=num_labels,
            return_dict=True,
            return_unused_kwargs=True,
        )

        self.test = (
            args_dict['test_run'] if args_dict['test_run'] is not None else False # default False
        )
        normalized_encoder, self.normalized_decoder, sorted_normalized_paths = self.intialize_hierarchy_paths()

        self.config.num_labels_per_lvl = self.tree_utils.get_number_of_nodes_lvl()
        self.config.paths = sorted_normalized_paths
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args_dict["checkpoint_tokenizer"], use_fast=True, model_max_length=512
        )

        if args_dict["task_type"] == "classification":
            # self.model = AutoModelForSequenceClassification.from_config(self.config)
            self.model = ClassificationModel(self.config)
        elif args_dict["task_type"] == "NER":
            self.model = AutoModelForTokenClassification.from_config(self.config)
        
        else:
            raise Exception(
                "Task unknown. Add the new task or use a kown model ('classification', 'NER')"
            )
        
        self.model.to(args_dict['device'])
        #print(self.model)

        tf_ds = {}
        for key in self.dataset:
            df_ds = self.dataset[key]
            if self.test:
                # load only subset of the data

                #DONT COMMIT CHANGE
                #df_ds = df_ds[df_ds['category'] == '67010100_Clothing Accessories']
                df_copy = df_ds
                df_copy.drop_duplicates(subset=['label'])
                test_run_label_num = (
                    args_dict['test_run_label_num'] if args_dict['test_run_label_num'] is not None else 20 # default to 20 labels
                )
                if len(df_copy) < test_run_label_num: test_run_label_num = len(df_copy)
                label_sample = list(df_copy.sample(n = test_run_label_num).label)

                df_ds = df_ds[df_ds['label'].isin(label_sample)]

            tf_ds[key] = CategoryDatasetFlat(df_ds, normalized_encoder)

        self.dataset = tf_ds
        if args_dict["freeze_layer"]:
            self.freeze(args_dict["freeze_layer"], model_typ=args_dict["architecture"])

    def load_tree(self):
        data_dir = Path('data')
        path_to_tree = data_dir.joinpath('tree_{}.pkl'.format(self.data_name))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

        self.tree_utils = TreeUtils(self.tree)

    def intialize_hierarchy_paths(self):
        """initialize paths using the provided tree"""
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree if node[1] == 0]
        paths = [self.tree_utils.determine_path_to_root([node]) for node in leaf_nodes]

        # Normalize paths per level in hierarchy - currently the nodes are of increasing number throughout the tree.
        normalized_paths = [self.tree_utils.normalize_path_from_root_per_level(path) for path in paths]

        normalized_encoder = {'Root': {'original_key': 0, 'derived_key': 0}}
        normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}

        #initiaize encoders
        for path, normalized_path in zip(paths, normalized_paths):
            key = path[-1]
            derived_key = normalized_path[-1]
            if key in leaf_nodes:
                normalized_encoder[decoder[key]] = {'original_key': key, 'derived_key': derived_key}
                normalized_decoder[derived_key] = {'original_key': key, 'value': decoder[key]}

        oov_path = [[0, 0, 0]]
        normalized_paths = oov_path + normalized_paths

        #Align length of paths if necessary
        longest_path = max([len(path) for path in normalized_paths])

        # Sort paths ascending
        sorted_normalized_paths = []
        for i in range(len(normalized_paths)):
            found_path = normalized_paths[0]
            for path in normalized_paths:
                for found_node, node in zip(found_path,path):
                    if found_node > node:
                        found_path = path
                        break

            if not (found_path is None):
                sorted_normalized_paths.append(found_path)
                normalized_paths.remove(found_path)

        return normalized_encoder, normalized_decoder, sorted_normalized_paths

    def freeze(self, n_freeze_layer: int, model_typ: str = "roberta"):
        """freezes the layer of the bert model and adds a unfrozen classifier at the end
        """
        # TODO: work in model.named_modules() to get all layers and make this code architecture independet

        # This if statement catches the different namings of layers in the model architecture 
        if model_typ == "roberta":
            model_arc = self.model.roberta

        elif model_typ == "bert":
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
                model_arc.encoder.layer,
                *model_arc.encoder.layer[: min(n_freeze_layer, max_bert_layer)], # Defines which layers are to be frozen according to the chosen hyperparameters
            ]
        else:  # Must be DistilBert, all unkown architectures have been filtered in the step before. In case a new architecure is added (apart from distil, roberta and bert) modify the if statement accordingly
            max_bert_layer = len(model_arc.transformer.layer)
            freezable_modules = [
                model_arc.transformer.layer,
                *model_arc.transformer.layer[: min(n_freeze_layer, max_bert_layer)], # Defines which layers are to be frozen according to the chosen hyperparameters
            ]

        # Default RoBERTa model "gottBERT" has 12 layers
        # Default DistilBERT has 6 layers
        for module in freezable_modules:
            for param in module.parameters():
                param.requires_grad = False # This means we do not want to change the weights of the layer throughout the training process

        for k, v in model_arc.named_parameters():
            if v.requires_grad:
                print("{}: {}".format(k, v.requires_grad)) # Prints a list of all layers that can still be trained after freezing

    def get_datasets(self):
        return self.dataset['train'], self.dataset['valid'], self.dataset['test']
    def get_tree(self):
        return self.tree
    def get_decoder(self):
        return self.normalized_decoder
