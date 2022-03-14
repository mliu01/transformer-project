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
        self.decoder = decoder
        self.tree = tree
        self.numeric_hierarchy = self.successors_per_node()

    def successors_per_node(self):
        '''Creates dictionary with every node and their child nodes
        '''#

        ## TODO: decode derived key 
        numeric_hierarchy = {}
        for i in self.tree.nodes():
            numeric_hierarchy[i] = list(self.tree.successors(i))

        return numeric_hierarchy


    def check_hierarchy(self, current_level, previous_level, lvl): 
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
        current_level = [self.decoder[lvl+1][i.item()] if i.item() in self.decoder[lvl+1] else False for i in current_level]
        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [not current_level[i] in self.numeric_hierarchy[self.decoder[lvl][previous_level[i].item()]] if previous_level[i].item() in self.decoder[lvl] else True
            for i in range(previous_level.size()[0])]

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