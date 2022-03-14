# %%
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
import networkx as nx


# %%
class RNNClassificationModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.model = AutoModel.from_config(self.config, add_pooling_layer=False)
        self.classifier = RNNHead(self.config)

        self.model.init_weights()      
        
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

        outputs = self.model(
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
        
        sequence_output = outputs[0][:, 0, :].view(-1, 768)  # take <s> token (equiv. to [CLS])

        if labels is not None:
            loss = None
            logits_list = []
            transposed_labels = torch.transpose(labels, 0, 1)
            hidden = self.classifier.initHidden(len(labels))
            # Initialize Head
            self.classifier.zero_grad()

            for i in range(len(transposed_labels)):
                logits_lvl, hidden = self.classifier(sequence_output, hidden)

                logits_list.append(logits_lvl)

                loss_fct = nn.CrossEntropyLoss()

                if loss is None:
                    loss = loss_fct(logits_lvl.view(-1, self.config.num_labels), transposed_labels[i].view(-1))
                else:
                    loss += loss_fct(logits_lvl.view(-1, self.config.num_labels), transposed_labels[i].view(-1))

            logits = torch.stack(logits_list)
            logits = torch.transpose(logits, 0, 1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save(self, model, optimizer, output_path):
        # save
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_path)       


class RNNHead(nn.Module):
    def __init__(self, config):
        super(RNNHead, self).__init__()

        self.hidden_size = config.hidden_size

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.i2h = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(config.hidden_size + config.hidden_size, config.num_labels)

        self.o2o = nn.Linear(config.hidden_size + config.num_labels, config.num_labels)

        # Not used for now!
        #self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = self.dropout(combined)

        hidden = self.i2h(combined)
        hidden = torch.tanh(hidden)

        output = self.i2o(combined)
        output = torch.tanh(output)

        output_combined = torch.cat((hidden, output), 1)
        output = self.dropout(output_combined)
        output = self.o2o(output)

        return output, hidden

    def initHidden(self, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, self.hidden_size).to(device)
        