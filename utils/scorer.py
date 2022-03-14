from sklearn.metrics import precision_recall_fscore_support, f1_score
from networkx import all_pairs_shortest_path_length, relabel_nodes
from contextlib import contextmanager
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import numpy as np
from datasets import load_metric

def score_mwpd(gs: list, prediction: list):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)
    macro_scores = precision_recall_fscore_support(gs, prediction, average='macro', zero_division=0)
    weighted_scores = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    return {'weighted': weighted_scores, 'macro': macro_scores}


def h_score(y_true, y_pred, class_hierarchy, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    y_true_ = fill_ancestors(y_true, root, graph=class_hierarchy)
    y_pred_ = fill_ancestors(y_pred, root, graph=class_hierarchy)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_positives = np.count_nonzero(y_true_)
    all_results = np.count_nonzero(y_pred_)

    h_precision = 0
    if all_results > 0:
        h_precision = true_positives / all_results

    h_recall = 0
    if all_positives > 0:
        h_recall = true_positives / all_positives

    return h_precision, h_recall


def h_fbeta_score(y_true, y_pred, class_hierarchy, root, beta=1.):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    hP, hR = h_score(y_true, y_pred, class_hierarchy, root)
    if (beta ** 2. * hP + hR) > 0:
        return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)
    else:
        return 0


@contextmanager
def multi_labeled(y_true, y_pred, graph, root):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    mlb = MultiLabelBinarizer()
    all_classes = [
        node
        for node in graph.nodes
        if node != root
    ]
    # print('all_classes',all_classes) Nb. we pass a (singleton) list-within-a-list as fit() expects an iterable of
    # iterables -> Changed implementation here
    all_classes_new = []
    for klasse in all_classes:
        all_classes_new.append([klasse])

    mlb.fit(all_classes_new)

    y_true_new = []
    for klasse in y_true:
        y_true_new.append([klasse])

    y_pred_new = []
    for klasse in y_pred:
        y_pred_new.append([klasse])

    node_label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(list(mlb.classes_))
    }

    yield (
        mlb.transform(y_true_new),
        mlb.transform(y_pred_new),
        relabel_nodes(graph, node_label_mapping),
        root,
    )


def hierarchical_score(y_true, y_pred, tree, root, name='Unknown', decoder=None):
    logger = logging.getLogger(__name__)

    # decode from derived to orig. key, since graph has orig keys only!
    level = list(decoder.keys())[-1]
    y_true = [decoder[level][l] for l in y_true]
    y_pred = [decoder[level][p] if p in decoder[level] else 0 for p in y_pred]


    with multi_labeled(y_true, y_pred, tree, root) as (y_test_, y_pred_, graph_, root_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
            root_,
        )
        if not name:
            return h_fbeta
        else:
            logger.info("{} - Hierarchy: | h_f1: {:4f}".format(name, h_fbeta))
            return h_fbeta


def fill_ancestors(y, root, graph, copy=True):
    """ Influenced by https://github.com/asitang/sklearn-hierarchical-classification/blob/develop
    /sklearn_hierarchical/metrics.py """
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        if target == root:
            # Our stub ROOT node, can skip
            continue
        ix_rows = np.where(y[:, target] > 0)[0]
        # all ancestors, except the last one which would be the root node
        ancestors = list(distances.keys())[:-1]
        y_[tuple(np.meshgrid(ix_rows, ancestors))] = 1
    graph.reverse(copy=False)
    return y_


class HierarchicalScorer:
    def __init__(self, experiment_name, tree, transformer_decoder=None):
        self.logger = logging.getLogger(__name__)

        self.experiment_name = experiment_name
        self.tree = tree
        self.transformer_decoder = transformer_decoder
        
        self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

        self.accuracy = load_metric("accuracy")
        self.matthews_correlation = load_metric("matthews_correlation")
        self.f1_score = load_metric("f1")


    def compute_metrics_transformers_hierarchy(self, pred, report=False):
        labels_per_lvl, preds_per_lvl = self.transpose_hierarchy(pred)

        if report:
            return labels_per_lvl, preds_per_lvl
        return self.compute_metrics(labels_per_lvl, preds_per_lvl)

    def transpose_hierarchy(self, pred):
        labels_paths = [list(label) for label in pred.label_ids]
        preds_paths = np.array([list(prediction.argmax(-1)) for prediction in pred.predictions]).transpose().tolist() #[list(prediction.argmax(-1)) for prediction in pred.predictions]

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = [list(prediction.argmax(-1)) for prediction in pred.predictions] #np.array(preds_paths).transpose().tolist()

        return labels_per_lvl, preds_per_lvl

    def compute_metrics(self, labels_per_lvl, preds_per_lvl):
        """Compute Metrics for leaf nodes and all nodes in the graph separately"""
        self.logger.debug('Leaf nodes')
        h_f_score = hierarchical_score(labels_per_lvl[-1], preds_per_lvl[-1], self.tree, self.root, name=self.experiment_name, decoder=self.transformer_decoder)

        results = {'h_f1': h_f_score}

        counter = 0

        sum_prec = {'weighted': 0.0, 'macro': 0.0}
        sum_rec = {'weighted': 0.0, 'macro': 0.0}
        sum_f1 = {'weighted': 0.0, 'macro': 0.0}
        sum_mc = 0.0
        sum_acc = 0.0

        for labels_lvl, preds_lvl in zip(labels_per_lvl, preds_per_lvl):

            counter += 1
            labels_lvl = [value for value in labels_lvl if value != -1]
            preds_lvl = [value for value in preds_lvl if value != -1]

            self.logger.debug('lvl_{}'.format(counter))
            score_dict = score_mwpd(labels_lvl, preds_lvl)
            acc = list(self.accuracy.compute(predictions=preds_lvl, references=labels_lvl).values())[0]
            m_c = list(self.matthews_correlation.compute(predictions=preds_lvl, references=labels_lvl).values())[0]

            for key in score_dict:
                prec, rec, f1, support = score_dict[key]
                results['{}_prec_lvl_{}'.format(key, counter)] = prec
                results['{}_rec_lvl_{}'.format(key, counter)] = rec
                results['{}_f1_lvl_{}'.format(key, counter)] = f1

                self.logger.info(
                    "{} - Lvl{}: | {}_prec: {:4f} | {}_rec: {:4f} | {}_f1: {:4f}".format(
                        self.experiment_name, counter, key, prec, key, rec, key, f1))

                sum_prec[key] += prec
                sum_rec[key] += rec
                sum_f1[key] += f1

            results['lvl_{}_acc'.format(counter)] = acc
            results['lvl_{}_matt_corr'.format(counter)] = m_c
            sum_acc += sum_acc
            sum_mc += m_c
            self.logger.info(
                    "{} - Lvl{}: | acc: {:4f} | matt_corr: {:4f}".format(
                        self.experiment_name, counter, acc, m_c))


        for key in sum_prec:

            avg_prec = sum_prec[key] / len(labels_per_lvl)
            avg_rec = sum_rec[key] / len(labels_per_lvl)
            avg_f1 = sum_f1[key] / len(labels_per_lvl)

            results['avg_{}_prec'.format(key)] = avg_prec
            results['avg_{}_rec'.format(key)] = avg_rec
            results['avg_{}_f1'.format(key)] = avg_f1

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
