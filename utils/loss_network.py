import torch
import torch.nn as nn
from torch.autograd import Variable


class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, tree, decoder, device='cpu', total_level=3, alpha=1, beta=0.8, p_loss=3):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device

        self.hierarchy = tree
        self.decoder = decoder


    def check_hierarchy(self, current_level, previous_level, l): 
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''

        # if current pred is a successor of prev pred -> 0; if not -> 1
        bool_tensor = [not self.decoder[l][current_level[i].item()]['original_key'] if current_level[i].item() != 0 else True in list(self.hierarchy.successors(self.decoder[l-1][previous_level[i].item()]['original_key'])) for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        loss_fct = nn.CrossEntropyLoss()

        lloss = 0
        for l in range(self.total_level):

            lloss += loss_fct(predictions[l], true_labels[l])

        return  self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred, l)

            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return Variable(self.beta * dloss, requires_grad=True)