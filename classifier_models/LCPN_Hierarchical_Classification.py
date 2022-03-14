# %%
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel


# %%
class LCPNClassificationModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = AutoModel.from_config(self.config, add_pooling_layer=False)
        self.classifier = LCPNClassificationHead(self.config)

        self.roberta.init_weights()      
        
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
        
        sequence_output = outputs[0][:, 0, :].view(-1, 768)  # take <s> token (equiv. to [CLS])

        loss = None
        logits = None
        if labels is not None:
            logits, loss = self.classifier(sequence_output, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,logits=logits,hidden_states=outputs.hidden_states,attentions=outputs.attentions)

    def save(self, model, optimizer, output_path):
        # save
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_path)       
    
class LCPNClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels 
        self.paths_per_lvl = self.initialize_paths_per_lvl(config.paths)

        self.num_labels_per_lvl = {}

        for count, number in enumerate(config.num_labels_per_lvl.items()):
            self.num_labels_per_lvl[count] = number[1]

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a weight matrix and bias vector for each node in the tree
        self.nodes = {}
        self.nodes = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.hidden_size, 1).to(self.device) for i in range(self.num_labels_per_lvl[lvl])])
            for lvl in self.num_labels_per_lvl
        ])
        
    def forward(self, input, labels):
        # Make a prediction for all nodes in the tree and full paths
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        logits = None

        input = self.dropout(input)

        # Make prediction for each lvl in hierarchy
        for lvl, lvl_node in enumerate(self.nodes):
            logit_list = [node(input) for node in lvl_node]
            logits = torch.stack(logit_list, dim=1).to(self.device)

            updated_labels = self.update_label_per_lvl(labels, lvl, True)

            if loss is None:
                loss = loss_fct(logits.view(-1, self.num_labels_per_lvl[lvl]), updated_labels.view(-1))
            else:
                loss += loss_fct(logits.view(-1, self.num_labels_per_lvl[lvl]), updated_labels.view(-1))


        #lvl = 3 --> longest path
        logit_list = [self.predict_along_path(input,path, len(self.paths_per_lvl)) for path in self.paths_per_lvl[len(self.paths_per_lvl)-1]]
        logits = torch.stack(logit_list, dim=1).to(self.device)

        updated_labels = self.update_label_per_lvl(labels, len(self.paths_per_lvl)-1, True)

        if loss is None:
            loss = loss_fct(logits.view(-1, self.num_labels_per_lvl[len(self.paths_per_lvl)-1]), updated_labels.view(-1)) #self.num_labels_per_lvl[3]
        else:
            loss += loss_fct(logits.view(-1, self.num_labels_per_lvl[len(self.paths_per_lvl)-1]), updated_labels.view(-1))

        #Return only logits of last run to receive only valid paths!
        return logits, loss

    def forward_along_paths(self, input, labels):
        # Make a prediction along all paths in the tree
        loss_fct = nn.CrossEntropyLoss()
        loss = None
        logits = None

        input = self.dropout(input)

        # Make prediction for each lvl in hierarchy along path to hierarchy lvl
        for lvl in self.paths_per_lvl:
        #lvl = 3
            logit_list = [self.predict_along_path(input,path, lvl) for path in self.paths_per_lvl[f"lvl_{lvl}"]]
            logits = torch.stack(logit_list, dim=1).to(self.device)

            updated_labels = self.update_label_per_lvl(labels, lvl)

            if loss is None:
                loss = loss_fct(logits.view(-1, self.num_labels_per_lvl[f"lvl_{lvl}"]), updated_labels.view(-1))
            else:
                loss += loss_fct(logits.view(-1, self.num_labels_per_lvl[f"lvl_{lvl}"]), updated_labels.view(-1))

        #Return only logits of last run to receive only valid paths!
        return logits, loss

    def predict_along_path(self, input, path, lvl):
        # Make predictions along path
        logits = [torch.sigmoid(self.nodes[i][path[i]](input)) for i in range(lvl)] #self.nodes[0][0](input)
        logits = torch.cat(logits, dim=1)

        # Calculate logit for given input and path
        logit = torch.prod(logits, dim=1)

        return logit

    def initialize_paths_per_lvl(self, paths):
        length = max([len(path) for path in paths])
        paths_per_lvl = {}
        for i in range(length):
            added_paths = set()
            paths_per_lvl[i] = []
            for path in paths:
                new_path = path[:i+1]
                new_tuple = tuple(new_path)
                if not (new_tuple in added_paths):
                    added_paths.add(new_tuple)
                    paths_per_lvl[i].append(new_path)

        return paths_per_lvl

    def update_label_per_lvl(self, labels, lvl, all_paths=False):

        unique_values = labels
        updated_labels = labels.clone()
        for value in unique_values:
            searched_path = self.paths_per_lvl[len(self.paths_per_lvl)-1][value]
            if len(searched_path) >= lvl:
                update_value = searched_path[lvl]
                updated_labels[updated_labels==value] = update_value

        if not all_paths:
            updated_labels = [path for path in updated_labels if len(path) == lvl]

        return updated_labels
