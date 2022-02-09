import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig
)
import inspect


# +
class ClassificationModel(nn.Module):
    def __init__(self, config):
        super(ClassificationModel, self).__init__()
        self.config = config
        self.roberta = AutoModel.from_config(self.config)
        self.classifier = FlatClassificationHead(self.config)
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        r""" from AutoModelSequenceClassification.forward
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {'loss': loss, 'logits': logits, 'hidden_states': outputs.hidden_states, 'attentions': outputs.attentions}
    
class FlatClassificationHead(nn.Module):
    def __init__(self, config):
        super(FlatClassificationHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
       
        self.act = nn.Sigmoid()
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        #self.softmax = nn.Softmax() #not needed if cross entropy loss is calculated
        
    def forward(self, inputs, **kwargs):
        #nn.Flatten()
        outputs = inputs[0]
        outputs = self.dropout(outputs)
        outputs = self.dense(outputs)
        outputs = self.act(outputs)
        
        outputs = self.dropout(outputs)
        outputs = self.out_proj(outputs)
        #outputs = self.softmax(outputs)
        
        return outputs

# +
#model = AutoModelForSequenceClassification.from_pretrained('uklfr/gottbert-base')

# +
#lines = inspect.getsource(model.classifier.forward)

# +
#print(lines)

# +
#ClassificationModel(config)

# +
#config, unused_kwargs = AutoConfig.from_pretrained(
#            'uklfr/gottbert-base',
#            output_attentions=False,
#            num_labels=3,
#            return_dict=True,
#            return_unused_kwargs=True
#        )
