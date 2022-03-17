import torch.nn as nn
from transformers import (
    AutoConfig,
    set_seed,
)
from utils.model_runner import provide_model_and_tokenizer
from utils.tree_utils import TreeUtils
from utils.category_dataset import CategoryDatasetFlat, CategoryDatasetRNN

from pathlib import Path
import pickle

class BERT(nn.Module):
    def __init__(
        self, args_dict: dict, device: str = "cuda", num_labels: int = 10, num_labels_per_lvl=None, dataset=None
    ):
        """Loads the model and freezes the layers

        Args:
            args_dict (dict): argparse dictonary
            model_name (str, optional): hugging face model to be used. Defaults to "uklfr/gottbert-base".
            tokenizer_checkpoint: same as above for tokenizer
            num_labels (int, optional): number of labels. 1 for Regression. Default to 10 classes.
        """
        super().__init__()
        self.args_dict = args_dict

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
            if args_dict["task_type"] == "hierarchical-classification":
                self.config.num_labels = num_labels
                self.config.num_labels_per_lvl = num_labels_per_lvl

            elif args_dict["task_type"] == "lcpn-hierarchical-classification":
                self.dataset = dataset
                self.tree = None
                self.load_tree()
                if self.tree == None:
                    raise Exception(
                        "No tree was loaded, please provide a directed graph for your dataset. (networkx.DiGraph)"
                    )
                normalized_encoder, self.normalized_decoder, sorted_normalized_paths = self.intialize_hierarchy_paths()

                self.config.paths = sorted_normalized_paths
                self.config.num_labels_per_lvl = self.tree_utils.get_number_of_nodes_lvl()

                tf_ds = {}
                for key in self.dataset:
                    tf_ds[key] = CategoryDatasetFlat(self.dataset[key], normalized_encoder)
                self.dataset = tf_ds
            
            elif args_dict["task_type"] == "rnn-hierarchical-classification":
                self.dataset = dataset
                self.tree = None
                self.load_tree()
                if self.tree == None:
                    raise Exception(
                        "No tree was loaded, please provide a directed graph for your dataset. (networkx.DiGraph)"
                    )
                normalized_encoder, self.normalized_decoder, number_of_labels = self.encode_labels()
                self.config.num_labels = number_of_labels

                tf_ds = {}
                for key in self.dataset:
                    tf_ds[key] = CategoryDatasetRNN(self.dataset[key], normalized_encoder) 
                self.dataset = tf_ds

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


    def load_tree(self):
        data_dir = Path(self.args_dict['data_folder'])
        path_to_tree = data_dir.joinpath('tree_{}.pkl'.format(self.args_dict['data_file'].split('.')[0]))

        with open(path_to_tree, 'rb') as f:
            self.tree = pickle.load(f)

        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

        self.tree_utils = TreeUtils(self.tree)

## Functions for LCPN
    def intialize_hierarchy_paths(self):
        """initialize paths using the provided tree"""

        leaf_nodes = [node[0] for node in self.tree.out_degree if node[1] == 0]
        paths = [self.tree_utils.determine_path_to_root([node]) for node in leaf_nodes]

        # Normalize paths per level in hierarchy - currently the nodes are of increasing number throughout the tree.
        normalized_paths = [self.tree_utils.normalize_path_from_root_per_level(path) for path in paths]

        normalized_encoder = {'Root': {'original_key': 0, 'derived_key': 0}}
        normalized_decoder = { 0: {'original_key': 0, 'value': 'Root'}}
        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

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

## Functions for RNN
    def determine_path_to_root(self, nodes):
        predecessors = self.tree.predecessors(nodes[-1])
        predecessor = [k for k in predecessors][0]

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def normalize_path_from_root_per_parent(self, path):
        """Normalize label values per parent node"""
        found_successor = self.root
        normalized_path = []
        for searched_successor in path:
            counter = 0
            successors = self.tree.successors(found_successor)
            for successor in successors:
                counter += 1
                if searched_successor == successor:
                    normalized_path.append(counter)
                    found_successor = searched_successor
                    break

        assert (len(path) == len(normalized_path))
        return normalized_path

    def encode_labels(self):
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
                path = self.determine_path_to_root([encoder[key]])
                path = self.normalize_path_from_root_per_parent(path)
                
                normalized_encoder[key] = {'original_key': encoder[key], 'derived_key': counter,
                                           'derived_path': path}
                normalized_decoder[counter] = {'original_key': encoder[key], 'value': key}
                if len(path) > longest_path:
                    longest_path = len(path)

                counter += 1

        #Align path length
        fill_up_category = len(self.tree)

        for key in normalized_encoder:
            while len(normalized_encoder[key]['derived_path']) < longest_path:
                normalized_encoder[key]['derived_path'].append(fill_up_category)

        # Total number of labels is determined by the number of labels in the tree + 1 for out of category
        number_of_labels = len(self.tree) + 1

        return normalized_encoder, normalized_decoder, number_of_labels

    def get_datasets(self):
        return self.dataset['train'], self.dataset['valid'], self.dataset['test']
        
    def get_tree(self):
        return self.tree

    def get_decoder(self):
        return self.normalized_decoder