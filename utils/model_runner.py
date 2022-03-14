from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from classifier_models.Hierarchical_Classification import HierarchicalClassificationModel
from classifier_models.LCPN_Hierarchical_Classification import LCPNClassificationModel
from classifier_models.RNN_Hierarchical_Classification import RNNClassificationModel


def provide_model_and_tokenizer(name, pretrained_model_or_path, config=None):

    if name:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_or_path, use_fast=True, model_max_length=512)
        model = get_model(name)(config, pretrained_model_or_path)
        return tokenizer, model

    raise ValueError('Unknown model name: {}!'.format(name))

def get_model(name):
    if name == 'flat-classification':
        return flat_classification

    elif name == 'hierarchical-classification':
        return deep_hierarchy_classification

    elif name == 'lcpn-hierarchical-classification':
        return lcpn_classification

    elif name == 'rnn-hierarchical-classification':
        return rnn_classification

    elif name == 'NER':
        return ner 

# all models
def flat_classification(config, pretrained_model_or_path):
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_or_path, config=config)

def deep_hierarchy_classification(config, pretrained_model_or_path):
    return HierarchicalClassificationModel(config=config)

def lcpn_classification(config, pretrained_model_or_path):
    return LCPNClassificationModel(config=config)

def rnn_classification(config, pretrained_model_or_path):
    return RNNClassificationModel(config=config)

def ner(config, pretrained_model_or_path):
    return AutoModelForTokenClassification.from_pretrained(pretrained_model_or_path, config=config)