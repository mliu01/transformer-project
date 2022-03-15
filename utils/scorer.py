# %%
from sklearn.metrics import precision_recall_fscore_support, f1_score
import logging
import numpy as np
from datasets import load_metric

# %%
# metrics specifically for hierarchical classification tasks
# see paper: https://arxiv.org/pdf/2112.06560.pdf
# see sc: https://gitlab.com/dacs-hpi/hiclass/-/blob/master/hiclass/metrics.py
from hiclass import metrics

# %%
def hierarchical_score(y_true, y_pred):
    # transpose labels and predictions, has to be in shape (n_samples, n_levels)
    y_true_ = np.array(y_true).transpose()
    y_pred_ = np.array(y_pred).transpose()

    h_prec = metrics.precision(y_true_, y_pred_)
    h_recall = metrics.recall(y_true_, y_pred_)
    h_f1 = metrics.f1(y_true_, y_pred_)

    return h_prec, h_recall, h_f1


# %%
def score_mwpd(gs: list, prediction: list):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)
    macro_scores = precision_recall_fscore_support(gs, prediction, average='macro', zero_division=0)
    weighted_scores = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    return {'weighted': weighted_scores, 'macro': macro_scores}

# %%
class HierarchicalScorer:
    def __init__(self, experiment_name):
        self.logger = logging.getLogger(__name__)

        self.experiment_name = experiment_name

        self.accuracy = load_metric("accuracy")
        self.matthews_correlation = load_metric("matthews_correlation")
        self.f1_score = load_metric("f1")

    def transpose_hierarchy(self, pred):
        labels_paths = [list(label) for label in pred.label_ids]

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = [list(prediction.argmax(-1)) for prediction in pred.predictions] 

        return labels_per_lvl, preds_per_lvl

    def compute_metrics(self, pred):
        """Compute hierarchical metrics and flat metrics for each hierarchy level"""
        self.logger.debug('Computing metrics')

        labels_per_lvl, preds_per_lvl = self.transpose_hierarchy(pred)

        h_prec, h_recall, h_f1 = hierarchical_score(y_true=labels_per_lvl, y_pred=preds_per_lvl)

        results = {
            'h_prec': h_prec,
            'h_recall': h_recall,
            'h_f1': h_f1
        
        }

        lvl_counter = 0

        sum_prec = {'weighted': 0.0, 'macro': 0.0}
        sum_rec = {'weighted': 0.0, 'macro': 0.0}
        sum_f1 = {'weighted': 0.0, 'macro': 0.0}
        sum_mc = 0.0
        sum_acc = 0.0

        # flat scores for each hierarchy level
        for labels_lvl, preds_lvl in zip(labels_per_lvl, preds_per_lvl):

            counter += 1
            labels_lvl = [value for value in labels_lvl if value != -1]
            preds_lvl = [value for value in preds_lvl if value != -1]

            self.logger.debug(f'lvl_{lvl_counter}')
            score_dict = score_mwpd(labels_lvl, preds_lvl)
            acc = list(self.accuracy.compute(predictions=preds_lvl, references=labels_lvl).values())[0]
            m_c = list(self.matthews_correlation.compute(predictions=preds_lvl, references=labels_lvl).values())[0]

            for key in score_dict:
                prec, rec, f1, support = score_dict[key]
                results[f'{key}_prec_lvl_{lvl_counter}'] = prec
                results[f'{key}_rec_lvl_{lvl_counter}'] = rec
                results[f'{key}_f1_lvl_{lvl_counter}'] = f1

                self.logger.info(
                    "{} - Lvl{}: | {}_prec: {:4f} | {}_rec: {:4f} | {}_f1: {:4f}".format(
                        self.experiment_name, lvl_counter, key, prec, key, rec, key, f1))

                sum_prec[key] += prec
                sum_rec[key] += rec
                sum_f1[key] += f1

            results[f'lvl_{lvl_counter}_acc'] = acc
            results[f'lvl_{lvl_counter}_matt_corr'] = m_c
            sum_acc += acc
            sum_mc += m_c
            self.logger.info(
                    "{} - Lvl{}: | acc: {:4f} | matt_corr: {:4f}".format(
                        self.experiment_name, lvl_counter, acc, m_c))


        for key in sum_prec:

            avg_prec = sum_prec[key] / len(labels_per_lvl)
            avg_rec = sum_rec[key] / len(labels_per_lvl)
            avg_f1 = sum_f1[key] / len(labels_per_lvl)

            results[f'avg_{key}_prec'] = avg_prec
            results[f'avg_{key}_rec'] = avg_rec
            results[f'avg_{key}_f1'] = avg_f1

            self.logger.info(
                "{} - {} avg: | prec: {:4f} | rec: {:4f} | f1: {:4f}".format(
                    self.experiment_name, key, avg_prec, avg_rec, avg_f1))

        
        avg_acc = sum_acc / len(labels_per_lvl)
        avg_mc = sum_mc / len(labels_per_lvl)

        results['avg_acc'] = avg_acc
        results['avg_matt_corr'] = avg_mc
        self.logger.info(
                "{} - avg: | avg_acc: {:4f} |matt_corr: {:4f}".format(
                    self.experiment_name, avg_acc, avg_mc))


        return results
