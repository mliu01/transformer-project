from xmlrpc.client import Boolean
from sklearn.metrics import classification_report
from datasets import load_metric
import datasets
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from typing import Any, Dict, List, Set, Optional, Tuple, Union

_DESCRIPTION = """
TODO
"""
_CITATION = """\
TODO
"""

# TODO: unsure about the type annotation for the second tuple element

class Metrics():
    """
    Class structure brings benefit of being able to pass the label names
    """
    def __init__(self, run:object = None) -> None:
        """
        Takes the list of label names (care for alphabetical order behaviour) and optionally a run reference for using neptune.ai
        """
        self.accuracy = load_metric("accuracy")
        self.f1_score = load_metric("f1")
        self.precision = load_metric("precision")
        self.recall = load_metric("recall")
        self.matthews_correlation = load_metric("matthews_correlation")
        #self.class_scores = MultiClassScores()
        #self.class_scores.get_names(label_names)
        
        if run:
            self.run = run
            self.use_netune = True
        else: self.use_netune = False

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Gets a tuple of the prediction and the acctual label and computes metrics(F1, Acc., Prec., Matthews-Corr., Class-Score)
        and returns does a dictonary {"F1-Score": float}
        """
        prob, labels = eval_pred
        predictions = np.argmax(prob, axis=1)
        metrics = {}
        metrics.update(
            self.f1_score.compute(predictions=predictions,
                            references=labels, average="weighted")
        )
        metrics.update(
            self.accuracy.compute(
            predictions=predictions, references=labels))
        metrics.update(
            self.precision.compute(predictions=predictions,
                            references=labels, average="weighted")
        )
        metrics.update(
            self.recall.compute(predictions=predictions,
                        references=labels, average="weighted")
        )
        metrics.update(
            self.matthews_correlation.compute(
                predictions=predictions, references=labels)
        )
        # metrics.update(
        #     self.class_scores.compute(
        #         predictions=predictions, references=labels#, label_names=target_names
        #     )
        # )

        if self.use_netune:
            for key, value in metrics.items():
                self.run[f"training/batch/{key}"].log(value)

        return metrics


class MultiClassScores(datasets.Metric):
    """TODO: Short description of my metric."""

    def get_names(self, label_names):

        self.label_names = label_names

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
        )

    def _compute(
        self, predictions: np.ndarray, references: np.ndarray#, label_names: list = None
    ):
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        precision, recall, fscore, support = score(references, predictions)
        if self.label_names is None:
            self.label_names = np.unique(references)
        res = {}
        res.update({f"z_precision_{i}": val for i,
                   val in zip(self.label_names, precision)})
        res.update({f"z_recall_{i}": val for i,
                   val in zip(self.label_names, recall)})
        res.update({f"z_f1_{i}": val for i, 
                   val in zip(self.label_names, fscore)})
        res.update({f"zz_support_{i}": val for i,
                   val in zip(self.label_names, support)})

        return res