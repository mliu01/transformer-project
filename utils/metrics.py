from sklearn.metrics import classification_report
from datasets import load_metric
import datasets
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from typing import List, Set, Dict, Tuple, Optional

_DESCRIPTION = """
TODO
"""
_CITATION = """\
TODO
"""

# TODO: unsure about the type annotation for the second tuple element


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Gets a tuple of the prediction and the acctual label and computes metrics(F1, Acc., Prec., Matthews-Corr., Class-Score)
    and returns does a dictonary {"F1-Score": float}
    """
    prob, labels = eval_pred
    predictions = np.argmax(prob, axis=1)
    metrics = {}
    metrics.update(
        f1_score.compute(predictions=predictions,
                         references=labels, average="macro")
    )
    metrics.update(accuracy.compute(
        predictions=predictions, references=labels))
    metrics.update(
        precision.compute(predictions=predictions,
                          references=labels, average="macro")
    )
    metrics.update(
        recall.compute(predictions=predictions,
                       references=labels, average="macro")
    )
    metrics.update(
        matthews_correlation.compute(
            predictions=predictions, references=labels)
    )
    metrics.update(
        class_scores.compute(
            predictions=predictions, references=labels, label_names=target_names
        )
    )
    return metrics


class MultiClassScores(datasets.Metric):
    """TODO: Short description of my metric."""

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
        self, predictions: np.ndarray, references: np.ndarray, label_names: list = None
    ):
        """Returns the scores"""
        # TODO: Compute the different scores of the metric
        precision, recall, fscore, support = score(references, predictions)
        if label_names is None:
            label_names = np.unique(references)
        res = {}
        res.update({f"z_precision_{i}": val for i,
                   val in zip(label_names, precision)})
        res.update({f"z_recall_{i}": val for i,
                   val in zip(label_names, recall)})
        res.update({f"z_f1_{i}": val for i, val in zip(label_names, fscore)})
        res.update({f"zz_support_{i}": val for i,
                   val in zip(label_names, support)})

        return res


accuracy = load_metric("accuracy")
f1_score = load_metric("f1")
precision = load_metric("precision")
recall = load_metric("recall")
matthews_correlation = load_metric("matthews_correlation")
class_scores = MultiClassScores()