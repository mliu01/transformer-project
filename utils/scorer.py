from sklearn.metrics import precision_recall_fscore_support, f1_score
import logging
import numpy as np
from hiclass import metrics


def score_traditional(gs: list, prediction: list, name='Unknown'):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)


    w_prec, w_rec, w_f1, support = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    f1_macro = f1_score(gs, prediction, average='macro')

    logger.info(
        "{} - Leaf Nodes: | prec_weighted: {:4f} | rec_weighted: {:4f} | f1_weighted: {:4f} | f1_macro: {:4f}".format(
            name, w_prec, w_rec, w_f1, f1_macro))

    return [w_prec, w_rec, w_f1, f1_macro]  # Precision_weighted, Recall_weighted, F1_weighted, F1_macro


def score_mwpd(gs: list, prediction: list):
    logger = logging.getLogger(__name__)
    logger.debug(gs)
    logger.debug(prediction)
    macro_scores = precision_recall_fscore_support(gs, prediction, average='macro', zero_division=0)
    weighted_scores = precision_recall_fscore_support(gs, prediction, average='weighted', zero_division=0)
    return {'weighted': weighted_scores, 'macro': macro_scores}



def hierarchical_score(y_true, y_pred):
    # transpose labels and predictions, has to be in shape (n_samples, n_levels)
    y_true_ = np.array(y_true).transpose()
    y_pred_ = np.array(y_pred).transpose()

    h_prec = metrics.precision(y_true_, y_pred_)
    h_recall = metrics.recall(y_true_, y_pred_)
    h_f1 = metrics.f1(y_true_, y_pred_)

    return h_prec, h_recall, h_f1


class HierarchicalScorer:
    def __init__(self, experiment_name, tree=None, transformer_decoder=None):
        self.logger = logging.getLogger(__name__)

        self.experiment_name = experiment_name
        self.tree = tree
        self.transformer_decoder = transformer_decoder

        if tree:
            self.root = [node[0] for node in self.tree.in_degree if node[1] == 0][0]

# functions for flat 
    def determine_path_to_root(self, nodes):
        predecessors = [k for k in self.tree.predecessors(nodes[-1])]
        if len(predecessors) > 0:
            predecessor = predecessors[0]
        else:
            predecessor = self.root #Return ooc path

        if predecessor == self.root:
            nodes.reverse()
            return nodes
        nodes.append(predecessor)
        return self.determine_path_to_root(nodes)

    def determine_label_preds_per_lvl(self, labels, preds):
        label_paths = [self.determine_path_to_root([label]) for label in labels]
        pred_paths = [self.determine_path_to_root([pred]) for pred in preds]

        dummy_label = len(self.tree) + 1  # Dummy label used to align length of prediction paths if they differ
        longest_path = max([len(path) for path in label_paths])
        for label_path, pred_path in zip(label_paths, pred_paths):
            # Align length of prediction paths
            if len(label_path) != len(pred_path):
                prediction_difference = len(label_path) - len(pred_path)
                if prediction_difference > 0:
                    while prediction_difference > 0:
                        pred_path.append(dummy_label)
                        prediction_difference -= 1
                else:
                    while prediction_difference < 0:
                        label_path.append(dummy_label)
                        prediction_difference += 1

            #Add dummy to all paths
            while len(label_path) < longest_path and len(pred_path) < longest_path:
                label_path.append(-1)
                pred_path.append(-1)

        #Transpose paths
        label_per_lvl = np.array(label_paths).transpose().tolist()
        preds_per_lvl = np.array(pred_paths).transpose().tolist()

        return label_per_lvl, preds_per_lvl

    def get_all_nodes_per_lvl(self, level):
        successors = self.tree.successors(self.root)
        while level > 0:
            next_lvl_succesors = []
            for successor in successors:
                next_lvl_succesors.extend(self.tree.successors(successor))
            successors = next_lvl_succesors
            level -= 1
        return successors



    def compute_metrics_transformers_lcpn(self, pred):
        raw_labels = pred.label_ids
        raw_preds = pred.predictions.argmax(-1)

        labels = [self.transformer_decoder[label]['value'] for label in raw_labels]
        preds = [self.transformer_decoder[pred]['value'] for pred in raw_preds]

        decoder = dict(self.tree.nodes(data="name"))
        encoder = dict([(value, key) for key, value in decoder.items()])

        # Encoder values to compute metrics
        pp_labels = [encoder[label] for label in labels]
        pp_preds = [encoder[pred] for pred in preds]

        labels_per_lvl, preds_per_lvl = self.determine_label_preds_per_lvl(pp_labels, pp_preds)

        return self.compute_metrics(labels_per_lvl, preds_per_lvl)


    def compute_metrics_transformers_dhc(self, pred):
        labels_paths = [list(label) for label in pred.label_ids]

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = [list(prediction.argmax(-1)) for prediction in pred.predictions] 

        return self.compute_metrics(labels_per_lvl, preds_per_lvl)

    def compute_metrics_transformers_rnn(self, pred):
        labels_per_lvl, preds_per_lvl = self.transpose_rnn(pred)

        return self.compute_metrics(labels_per_lvl, preds_per_lvl)

    def transpose_rnn(self, pred):
        labels_paths = [list(label) for label in pred.label_ids]
        preds_paths = [list(prediction.argmax(-1)) for prediction in pred.predictions]

        labels_per_lvl = np.array(labels_paths).transpose().tolist()
        preds_per_lvl = np.array(preds_paths).transpose().tolist()

        return labels_per_lvl, preds_per_lvl

    def compute_metrics(self, labels_per_lvl, preds_per_lvl):
        """Compute Metrics for leaf nodes and all nodes in the graph separately"""
        self.logger.debug('Leaf nodes')
        h_prec, h_recall, h_f1 = hierarchical_score(labels_per_lvl, preds_per_lvl)

        results = { 
            'h_prec' :h_prec,
            'h_recall': h_recall,
            'h_f1': h_f1
            }

        counter = 0

        sum_prec = {'weighted': 0.0, 'macro': 0.0}
        sum_rec = {'weighted': 0.0, 'macro': 0.0}
        sum_f1 = {'weighted': 0.0, 'macro': 0.0}


        for labels_lvl, preds_lvl in zip(labels_per_lvl, preds_per_lvl):

            counter += 1
            labels_lvl = [value for value in labels_lvl if value != -1]
            preds_lvl = [value for value in preds_lvl if value != -1]

            self.logger.debug('lvl_{}'.format(counter))
            score_dict = score_mwpd(labels_lvl, preds_lvl)

            for key in score_dict:
                prec, rec, f1, support = score_dict[key]
                results['{}_prec_lvl_{}'.format(key, counter)] = prec
                results['{}_rec_lvl_{}'.format(key, counter)] = rec
                results['{}_f1_lvl_{}'.format(key, counter)] = f1

                self.logger.info(
                    "{} - Lvl{}: | {}_prec: {:4f} | {}_rec: {:4f} | {}_f1: {:4f}".format(
                        self.experiment_name, counter, key, prec, key, rec, key, f1 ))

                sum_prec[key] += prec
                sum_rec[key] += rec
                sum_f1[key] += f1


        for key in sum_prec:

            avg_prec = sum_prec[key] / len(labels_per_lvl)
            avg_rec = sum_rec[key] / len(labels_per_lvl)
            avg_f1 = sum_f1[key] / len(labels_per_lvl)

            results['average_{}_prec'.format(key)] = avg_prec
            results['average_{}_rec'.format(key)] = avg_rec
            results['average_{}_f1'.format(key)] = avg_f1

            self.logger.info(
                "{} - {} average: | prec: {:4f} | rec: {:4f} | f1: {:4f}".format(
                    self.experiment_name, key, avg_prec, avg_rec, avg_f1))

        return results
