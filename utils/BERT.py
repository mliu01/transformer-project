from email.policy import default
import torch.nn as nn
from transformers import (
    AutoConfig,
    set_seed,
)
from utils.model_runner import provide_model
from utils.tree_utils import TreeUtils
from utils.category_dataset import CategoryDataset

from pathlib import Path
import pickle
import networkx as nx

class BERT(nn.Module):
    def __init__(
        self, args_dict: dict, device: str = "cuda", num_labels=None, dataset=None
    ):
        """Loads the model and freezes the layers

        Args:
            args_dict (dict): argparse dictonary
            model_name (str, optional): hugging face model to be used. Defaults to "uklfr/gottbert-base".
            tokenizer_checkpoint: same as above for tokenizer
            num_labels (int, optional): number of labels. 1 for Regression. Default to 10 classes.
        """
        super().__init__()
        self.dataset = dataset
        self.tree = None
        self.load_tree(path=args_dict['data_folder'], name=args_dict['data_file'])
        if self.tree == None:
            raise Exception(
                "No tree was loaded, please provide a directed graph for your dataset. (networkx.DiGraph)"
            )

        self.config, self.unused_kwargs = AutoConfig.from_pretrained(
            args_dict["checkpoint_model_or_path"],
            hidden_dropout_prob=args_dict['hidden_dropout_prob'],
            attention_probs_dropout_prob=args_dict['attention_probs_dropout_prob'],
            output_attentions=False,
            num_labels=num_labels,
            return_dict=True,
            return_unused_kwargs=True,
        )

        if args_dict["task_type"] and args_dict["checkpoint_model_or_path"]:

            if args_dict["task_type"] == "lcpn-hierarchical-classification":
                normalized_encoder, self.normalized_decoder, sorted_normalized_paths = self.lcpn_encoder_decoder()

                self.config.paths = sorted_normalized_paths
                self.config.num_labels_per_lvl = self.tree_utils.get_number_of_nodes_lvl()        

            if args_dict['task_type'] == 'dhc-hierarchical-classification':
                normalized_encoder, self.normalized_decoder, num_labels = self.encode_labels()
                self.config.num_labels_per_lvl = self.tree_utils.get_number_of_nodes_lvl()
                self.config.decoder = self.normalized_decoder
            
            elif args_dict["task_type"] == "rnn-hierarchical-classification":
                normalized_encoder, self.normalized_decoder, number_of_labels = self.encode_labels()
                self.config.num_labels = number_of_labels


            # encoding dataset for hierarchical classification
            if not args_dict["task_type"] == "flat-classification":
                tf_ds = {}
                for key in self.dataset:
                    tf_ds[key] = CategoryDataset(self.dataset[key], normalized_encoder) 
                self.dataset = tf_ds

            self.model = provide_model(args_dict["task_type"], args_dict["checkpoint_model_or_path"], self.config, self.tree)
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
        model_arc_full = self.model

        # hierarchical classifiers can use any model (AutoModel)
        if not task_type == 'flat-classification':
            model_arc = self.model.model

        else:

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


    def load_tree(self, path, name):
        data_dir = Path(path)
        path_to_tree = data_dir.joinpath('tree_{}.pkl'.format(name.split('.')[0]))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

        self.tree_utils = TreeUtils(self.tree)

## Functions for LCPN
    def lcpn_encoder_decoder(self):
        """initialize paths using the provided tree"""

        leaf_nodes = [node[0] for node in self.tree.out_degree if node[1] == 0]
        paths = [self.tree_utils.determine_path_to_root([node]) for node in leaf_nodes]

        # Normalize paths per level in hierarchy - currently the nodes are of increasing number throughout the tree.
        normalized_paths = [self.tree_utils.normalize_path_from_root_per_level(path) for path in paths]

        normalized_encoder = {'Root': {'original_key': 0, 'derived': 0}}
        normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}
        decoder = dict(self.tree.nodes(data="name"))

        #initiaize encoders
        for path, normalized_path in zip(paths, normalized_paths):
            key = path[-1]
            derived_key = normalized_path[-1]
            if key in leaf_nodes:
                normalized_encoder[decoder[key]] = {'original_key': key, 'derived': derived_key}
                normalized_decoder[derived_key] = {'original_key': key, 'value': decoder[key]}

        oov_path = [[0, 0, 0]]
        normalized_paths = oov_path + normalized_paths

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

## Functions for DHC
    def encode_labels(self):
        """Provides encoder & decoder for labels on each hierarchy level"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]

        # building decoder that can decode any normalized label from each level
        longest_path = nx.dag_longest_path_length(self.tree) # tree depth
        all_edges = list(self.tree.edges())
        nodes_in_lvl = {0: [node[1] for node in all_edges if node[0]==0]} # what nodes are in level x?

        for level in range(1, longest_path):
            print(level)
            nodes_in_lvl[level] = [node[1] for node in all_edges if node[0] in nodes_in_lvl[level-1]]

        normalized_decoder = dict([(lvl, {}) for lvl in range(longest_path)]) # placeholder key for each level
        normalized_encoder = dict([(lvl, {}) for lvl in range(longest_path)])
        for key, val in nodes_in_lvl.items():
            counter = 1
            for i in val:
                normalized_decoder[key][counter] = {'original_key': i, 'value': decoder[i]}
                normalized_encoder[key][i] = {'derived_key': counter, 'value': decoder[i]}
                counter += 1

        # takes label (leaf node) and encodes it as normalized path
        label_encoder = {}
        counter = 0
        for key in encoder:
            if key in leaf_nodes:
                path = self.tree_utils.determine_path_to_root([encoder[key]])
                path = [normalized_encoder[counter][i]['derived_key'] for counter, i in enumerate(path)]
                label_encoder[key] = {'original_key': encoder[key], 'derived': path}

        number_of_labels = len(self.tree) + 1

        return label_encoder, normalized_decoder, number_of_labels

    def rnn_encoder_deocder(self):
        """Encode & decode labels plus rescale encoded values"""
        normalized_encoder = {}
        normalized_decoder = {}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        leaf_nodes = [node[0] for node in self.tree.out_degree(self.tree.nodes()) if node[1] == 0]
        leaf_nodes = [decoder[node] for node in leaf_nodes]

        counter = 0
        longest_path = 0
        for key in encoder:
            if key in leaf_nodes:
                path = self.tree_utils.determine_path_to_root([encoder[key]])
                path = self.tree_utils.normalize_path_from_root_per_parent(path)
                
                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter,
                                           'derived': path} #derived path
                normalized_decoder[tuple(path)] = {'original_key': encoder[key], 'value': key}
                if len(path) > longest_path:
                    longest_path = len(path)

                counter += 1

        #Align path length
        fill_up_category = len(self.tree)

        for key in normalized_encoder:
            while len(normalized_encoder[key]['derived']) < longest_path:
                normalized_encoder[key]['derived'].append(fill_up_category)

        # Total number of labels is determined by the number of labels in the tree + 1 for out of category
        number_of_labels = len(self.tree) + 1

        return normalized_encoder, normalized_decoder, number_of_labels


    def get_datasets(self):
        return self.dataset['train'], self.dataset['valid'], self.dataset['test']
        
    def get_tree(self):
        return self.tree

    def get_decoder(self):
        return self.normalized_decoder