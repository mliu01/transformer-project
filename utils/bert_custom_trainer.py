import torch
from transformers import Trainer
from torch import nn

from utils.loss_network import HierarchicalLossNetwork
from utils.tree_utils import TreeUtils
from utils.configuration import yaml_dump_for_notebook

class TrainerLossNetwork(Trainer):
    """
    Creates a custom Trainer class inheriting from transformers Trainer
    Computes dependency loss and layer loss
    ( based on "Deep Hierarchical Classification for Category Prediction in E-commerce System 2020" paper)

    Args:
        Takes all arguments used in the Huggingface Trainer class
        refer to https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer for documentation
    """

    def __init__(self, *args, **kwargs, ):
        super().__init__(*args, **kwargs)
        self.args_dict = yaml_dump_for_notebook(filepath='configs/hierarchical-baseline.yml')

        self.tree_utils = TreeUtils(self.args_dict['data_folder'], self.args_dict['data_file'])
        self.tree = self.tree_utils.tree

        self.normalized_decoder = self.tree_utils.encoding(only_decoder=True)

        self.HLN = HierarchicalLossNetwork(tree=self.tree, decoder=self.normalized_decoder, device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu'))


    def compute_loss(
        self, model: nn.Module, inputs: torch.Tensor, return_outputs=False
    ):

        labels = inputs['labels']
        outputs = model(**inputs)
        logits = outputs.logits
        transposed_labels = torch.transpose(labels, 0, 1)
        
        dloss = self.HLN.calculate_dloss(logits, transposed_labels)
        lloss = self.HLN.calculate_lloss(logits, transposed_labels)

        total_loss = lloss + dloss 

        return (total_loss, outputs) if return_outputs else total_loss


class SelfAdjDiceLoss(torch.nn.Module):
    """
    Taken from https://github.com/fursovia/self-adj-dice/blob/master/sadice/loss.py

    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)
    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(
        self, alpha: float = 1.0, gamma: float = 1.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.gamma) / (
            probs_with_factor + 1 + self.gamma
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss

        raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")


class TrainerDiceLoss(Trainer):
    """
    Creates a custom Trainer class inheriting from transformers Trainer
    to feed the implementation of the multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        Takes all arguments used in the Huggingface Trainer class
        refer to https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer for documentation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model: nn.Module, inputs: torch.Tensor, return_outputs=False
    ):
        """
        Overwritting Trainer.compute_loss to feed the SelfAdjDiceLoss accordingly
        For Docu and original source code refer to:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.compute_loss
        https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = SelfAdjDiceLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels)
        return (loss, outputs) if return_outputs else loss