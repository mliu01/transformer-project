# %%
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
import networkx as nx


# %%
class HierarchicalClassificationModel(PreTrainedModel):
    ''' Based on "Deep Hierarchical Classification for Category Prediction in E-commerce Systems"
        see: https://github.com/Ugenteraan/Deep_Hierarchical_Classification
    '''

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.model = AutoModel.from_config(self.config, add_pooling_layer=False)
        self.classifier = HierarchicalClassificationHead(self.config)

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
            return_dict=return_dict
        )
        
        sequence_output = outputs[0][:, 0, :].view(-1, 768)  # take <s> token (equiv. to [CLS])
        
        loss = None

        hidden = self.classifier.initHidden(len(labels))
        lvl1, lvl2, lvl3 = self.classifier(sequence_output, hidden)
        logits = [lvl1, lvl2, lvl3]

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


class HierarchicalClassificationHead(nn.Module):
    def __init__(self, config):
        super(HierarchicalClassificationHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_labels_per_lvl = config.num_labels_per_lvl

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear_lvl1 = nn.Linear(config.hidden_size + config.hidden_size, self.num_labels_per_lvl[0])
        self.linear_lvl2 = nn.Linear(config.hidden_size + config.hidden_size, self.num_labels_per_lvl[1])
        self.linear_lvl3 = nn.Linear(config.hidden_size + config.hidden_size, self.num_labels_per_lvl[2])

        self.softmax_reg1 = nn.Linear(self.num_labels_per_lvl[0], self.num_labels_per_lvl[0])
        self.softmax_reg2 = nn.Linear(self.num_labels_per_lvl[0] + self.num_labels_per_lvl[1], self.num_labels_per_lvl[1])
        self.softmax_reg3 = nn.Linear(self.num_labels_per_lvl[1] + self.num_labels_per_lvl[2], self.num_labels_per_lvl[2])



    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        combined = torch.tanh(combined)
        x = self.dropout(combined)

        level_1 = self.softmax_reg1(self.linear_lvl1(x))
        level_2 = self.softmax_reg2(torch.cat([level_1, self.linear_lvl2(x)], dim=1))
        level_3 = self.softmax_reg3(torch.cat((level_2, self.linear_lvl3(x)), dim=1))

        return level_1, level_2, level_3

    def initHidden(self, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, self.hidden_size).to(device)
        