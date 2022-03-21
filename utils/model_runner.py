from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from classifier_models.DHC_Hierarchical_Classification import DHCClassificationModel
from classifier_models.LCPN_Hierarchical_Classification import LCPNClassificationModel
from classifier_models.RNN_Hierarchical_Classification import RNNClassificationModel


def provide_model(name, pretrained_model_or_path, config=None, tree=None):

    if name == 'flat-classification':
        return AutoModelForSequenceClassification.from_pretrained(pretrained_model_or_path, config=config)

    elif name == 'dhc-hierarchical-classification':
        return DHCClassificationModel(config=config, tree=tree)

    elif name == 'lcpn-hierarchical-classification':
        return LCPNClassificationModel(config=config)

    elif name == 'rnn-hierarchical-classification':
        return RNNClassificationModel(config=config)

    elif name == 'NER':
        return AutoModelForTokenClassification.from_pretrained(pretrained_model_or_path, config=config)

    else:
        raise ValueError('Unknown model name: {}!'.format(name))
