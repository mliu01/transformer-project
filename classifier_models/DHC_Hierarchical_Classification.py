# %%
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from utils.loss_network import HierarchicalLossNetwork as HLN


# %%
class DHCClassificationModel(PreTrainedModel):
    ''' Based on "Deep Hierarchical Classification for Category Prediction in E-commerce Systems"
        see: https://github.com/Ugenteraan/Deep_Hierarchical_Classification
    '''

    def __init__(self, config, tree):
        super().__init__(config)
        self.config = config
        
        self.model = AutoModel.from_config(self.config, add_pooling_layer=False)
        self.classifier = DHCClassificationHead(self.config)

        self.HLN = HLN(tree, config.decoder, device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu'))

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

        #hidden = self.classifier.initHidden(len(labels))
        lvl1, lvl2, lvl3 = self.classifier(sequence_output)
        logits = [lvl1, lvl2, lvl3]


        transposed_labels = torch.transpose(labels, 0, 1) # from [lvl1, lvl2, lvl3] -> [lvl1], [lvl2], [lvl3]
        
        dloss = self.HLN.calculate_dloss(logits, transposed_labels)
        lloss = self.HLN.calculate_lloss(logits, transposed_labels)

        loss = lloss + dloss 

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  


class DHCClassificationHead(nn.Module):
    def __init__(self, config):
        super(DHCClassificationHead, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_labels_per_lvl = config.num_labels_per_lvl

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)

        self.linear_lvl1 = nn.Linear(config.hidden_size, self.num_labels_per_lvl[1])
        self.linear_lvl2 = nn.Linear(config.hidden_size, self.num_labels_per_lvl[2])
        self.linear_lvl3 = nn.Linear(config.hidden_size, self.num_labels_per_lvl[3])

        self.softmax_reg1 = nn.Linear(self.num_labels_per_lvl[1], self.num_labels_per_lvl[1])
        self.softmax_reg2 = nn.Linear(self.num_labels_per_lvl[1] + self.num_labels_per_lvl[2], self.num_labels_per_lvl[2])
        self.softmax_reg3 = nn.Linear(self.num_labels_per_lvl[2] + self.num_labels_per_lvl[3], self.num_labels_per_lvl[3])



    def forward(self, input):
        x = self.dense(input)
        x = torch.tanh(x)
        x0 = self.dropout(x)

        layer1_rep = nn.functional.relu(self.linear_lvl1(x0))
        layer2_rep = nn.functional.relu(self.linear_lvl2(x0))
        layer3_rep = nn.functional.relu(self.linear_lvl3(x0))

        level_1 = self.softmax_reg1(layer1_rep)
        level_2 = self.softmax_reg2(torch.cat((layer1_rep, layer2_rep), dim=1))
        level_3 = self.softmax_reg3(torch.cat((layer2_rep, layer3_rep), dim=1))

        return level_1, level_2, level_3

    def initHidden(self, size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.zeros(size, self.hidden_size).to(device)
        