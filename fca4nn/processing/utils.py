import pandas as pd
import json
import numpy as np
import pickle
import torch
import os
from collections import defaultdict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    hamming_loss,
    roc_auc_score,
)
import networkx as nx
import copy

from processing.imagenet10_classes import IMAGENET100_CLASS2ID
from collections import defaultdict

NUM_CLASSES = 100


# Also taken from the official CBM codebase
def create_new_dataset(
    out_dir, field_change, compute_fn, datasets=["train", "val", "test"], data_dir=""
):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mask = np.array(
        [
            1,
            4,
            6,
            7,
            10,
            14,
            15,
            20,
            21,
            23,
            25,
            29,
            30,
            35,
            36,
            38,
            40,
            44,
            45,
            50,
            51,
            53,
            54,
            56,
            57,
            59,
            63,
            64,
            69,
            70,
            72,
            75,
            80,
            84,
            90,
            91,
            93,
            99,
            101,
            106,
            110,
            111,
            116,
            117,
            119,
            125,
            126,
            131,
            132,
            134,
            145,
            149,
            151,
            152,
            153,
            157,
            158,
            163,
            164,
            168,
            172,
            178,
            179,
            181,
            183,
            187,
            188,
            193,
            194,
            196,
            198,
            202,
            203,
            208,
            209,
            211,
            212,
            213,
            218,
            220,
            221,
            225,
            235,
            236,
            238,
            239,
            240,
            242,
            243,
            244,
            249,
            253,
            254,
            259,
            260,
            262,
            268,
            274,
            277,
            283,
            289,
            292,
            293,
            294,
            298,
            299,
            304,
            305,
            308,
            309,
            310,
            311,
        ]
    )

    for dataset in datasets:
        path = os.path.join(data_dir, dataset + ".pkl")
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, "rb"))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)

            if field_change in new_d:
                old_value = d[field_change]
                new_value = list((np.array(old_value)[mask - 1]))
                assert type(old_value) == type(new_value)
            new_d[field_change] = new_value
            new_data.append(new_d)

        f = open(os.path.join(out_dir, dataset + ".pkl"), "wb")
        pickle.dump(new_data, f)
        f.close()


def get_info_from_lattice(lattice_path, lattice_levels, make_exclusive=False):
    def toposort_lattice(lattice):
        G = nx.DiGraph()
        for concept in lattice:
            G.add_node(concept)
            for upper in concept.upper_neighbors:
                G.add_edge(concept, upper)
        sorted_concepts = list(nx.topological_sort(G))
        return sorted_concepts

    def compute_levels(concepts):
        level = {}
        for concept in concepts:
            extent, intent = concept
            if str(extent) not in level:
                level[str(extent)] = 0
            for upper in concept.upper_neighbors:
                upper_extent, upper_intent = upper
                if str(upper_extent) not in level:
                    level[str(upper_extent)] = level[str(extent)] + 1
                else:
                    level[str(upper_extent)] = max(
                        level[str(upper_extent)], level[str(extent)] + 1
                    )
        return level

    def compute_hierarchy(concepts, level):
        hierarchy = {}
        for concept in concepts:
            extent, intent = concept
            lvl = level[str(extent)]
            if lvl in hierarchy:
                hierarchy[lvl].append(concept)
            else:
                hierarchy[lvl] = [concept]
        return hierarchy

    lattice_levels = sorted(lattice_levels, reverse=True)
    file_extension = os.path.splitext(lattice_path)[1]

    perlevel_intents = [[] for _ in range(len(lattice_levels))]
    perlevel_fcs = [[] for _ in range(len(lattice_levels))]

    if file_extension == ".json":
        with open(lattice_path, "r") as f:
            lattice = json.load(f)
        maxlevel = int(max([int(i) for i in lattice["hierarchy"].keys()]))
        print("maxlevel: ", maxlevel)

        for fc_id in lattice["hierarchy"][str(maxlevel - 3)].keys():
            l1.extend(lattice["hierarchy"][str(maxlevel - 3)][str(fc_id)]["intent"])
        for fc_id in lattice["hierarchy"]["3"].keys():
            l2.extend(lattice["hierarchy"]["3"][str(fc_id)]["intent"])

        l1 = list(set(l1))
        l2 = list(set(l2))

    elif file_extension == ".pkl":
        with open(lattice_path, "rb") as pkl:
            context = pickle.load(pkl)
            lattice = context.lattice

        sorted_concepts = toposort_lattice(lattice)
        level = compute_levels(sorted_concepts)
        hierarchy = compute_hierarchy(sorted_concepts, level)

        for i in range(len(lattice_levels)):
            # print(lattice_levels[i])
            for concept in hierarchy[lattice_levels[i]]:
                perlevel_intents[i].extend(concept.intent)
                perlevel_fcs[i].append(concept)
            perlevel_intents[i] = set(perlevel_intents[i])
        
        # Making the attribute sets exclusive
        if make_exclusive:
            for i, attrs in enumerate(perlevel_intents[1:], start=1):
                for j in range(0, i):
                    perlevel_intents[i] = perlevel_intents[i].difference(perlevel_intents[j])

        # Make everything a list
        for i in range(len(perlevel_intents)):
            perlevel_intents[i] = list(perlevel_intents[i])
            print(len(perlevel_intents[i]))
    return perlevel_intents, perlevel_fcs


class MultiLabelMetrics:
    def __init__(self):
        self.y_true_list = []
        self.y_pred_list = []

    def update(self, y_true, y_pred):
        """
        Updates the internal storage with a new batch of predictions and ground truth.
        :param y_true: np.array of shape (batch_size, num_labels)
        :param y_pred: np.array of shape (batch_size, num_labels)
        """
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)

    def compute(self):
        """
        Computes the final metrics over all stored batches.
        :return: Dictionary containing final metric values.
        """
        y_true = np.vstack(self.y_true_list)
        y_pred = np.vstack(self.y_pred_list)

        metrics = {
            "Hamming Loss": hamming_loss(y_true, y_pred),
            "Subset Accuracy": np.mean(
                np.all(y_true == y_pred, axis=1)
            ),  # Exact match ratio
            "Precision (Macro)": precision_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Recall (Macro)": recall_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "F1 Score (Macro)": f1_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Jaccard Index (Macro)": jaccard_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Accuracy of 1s": np.mean(
                np.sum(y_true * y_pred, axis=1) / np.sum(y_true, axis=1)
            ),
            "Accuracy of 0s": np.mean(
                np.sum((1 - y_true) * (1 - y_pred), axis=1)
                / np.sum((1 - y_true), axis=1)
            ),
        }

        # Compute MCC only if we have more than one label
        if y_true.shape[1] > 1:
            mcc_values = [
                matthews_corrcoef(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]
            metrics["MCC (Mean)"] = np.mean(mcc_values)

        return metrics


class MetricCalculator:
    def __init__(self, num_clfs):
        self.num_clfs = num_clfs
        self.cls_01_common = {"0": [0] * (num_clfs - 1), "1": [0] * (num_clfs - 1)}
        self.cls_01_correct = {"0": [0] * (num_clfs - 1), "1": [0] * (num_clfs - 1)}
        self.attr_01_common = {"0": [0] * num_clfs, "1": [0] * num_clfs}
        self.attr_01_correct = {"0": [0] * num_clfs, "1": [0] * num_clfs}

        # For AUC calculation
        self.cls_scores = [[] for _ in range(num_clfs - 1)]
        self.cls_labels = [[] for _ in range(num_clfs - 1)]
        self.attr_scores = [[] for _ in range(num_clfs)]
        self.attr_labels = [[] for _ in range(num_clfs)]

    def reset(self):
        self.cls_01_common = {
            "0": [0] * (self.num_clfs - 1),
            "1": [0] * (self.num_clfs - 1),
        }
        self.cls_01_correct = {
            "0": [0] * (self.num_clfs - 1),
            "1": [0] * (self.num_clfs - 1),
        }
        self.attr_01_common = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}
        self.attr_01_correct = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}

        self.cls_scores = [[] for _ in range(self.num_clfs - 1)]
        self.cls_labels = [[] for _ in range(self.num_clfs - 1)]
        self.attr_scores = [[] for _ in range(self.num_clfs)]
        self.attr_labels = [[] for _ in range(self.num_clfs)]

    def update(self, cls_preds, attr_preds, classes_present, attrs_present):
        for i in range(self.num_clfs - 1):
            pred_i = torch.round(cls_preds[i]).cpu()
            label_i = classes_present[i]

            self.cls_01_common["0"][i] += torch.sum((1 - pred_i) * (1 - label_i)).item()
            self.cls_01_correct["0"][i] += torch.sum(1 - label_i).item()
            self.cls_01_common["1"][i] += torch.sum(pred_i * label_i).item()
            self.cls_01_correct["1"][i] += torch.sum(label_i).item()

            self.cls_scores[i].extend(cls_preds[i].detach().cpu().numpy().tolist())
            self.cls_labels[i].extend(label_i.detach().cpu().numpy().tolist())

            # attribute update for i-th
            attr_pred_i = torch.round(attr_preds[i]).cpu()
            attr_label_i = attrs_present[i]

            self.attr_01_common["0"][i] += torch.sum(
                (1 - attr_pred_i) * (1 - attr_label_i)
            ).item()
            self.attr_01_correct["0"][i] += torch.sum(1 - attr_label_i).item()
            self.attr_01_common["1"][i] += torch.sum(attr_pred_i * attr_label_i).item()
            self.attr_01_correct["1"][i] += torch.sum(attr_label_i).item()

            self.attr_scores[i].extend(attr_preds[i].detach().cpu().numpy().tolist())
            self.attr_labels[i].extend(attr_label_i.detach().cpu().numpy().tolist())

        # last attribute (final)
        self.attr_01_common["0"][-1] += torch.sum(
            (1 - torch.round(attr_preds[-1]).cpu()) * (1 - attrs_present[-1])
        ).item()
        self.attr_01_correct["0"][-1] += torch.sum(1 - attrs_present[-1]).item()
        self.attr_01_common["1"][-1] += torch.sum(
            torch.round(attr_preds[-1]).cpu() * attrs_present[-1]
        ).item()
        self.attr_01_correct["1"][-1] += torch.sum(attrs_present[-1]).item()

        self.attr_scores[-1].extend(attr_preds[-1].detach().cpu().numpy().tolist())
        self.attr_labels[-1].extend(attrs_present[-1].detach().cpu().numpy().tolist())

    def calculate_accuracy(self):
        cls_01_acc = {"0": [0] * (self.num_clfs - 1), "1": [0] * (self.num_clfs - 1)}
        attr_01_acc = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}
        for i in range(self.num_clfs - 1):
            cls_01_acc["0"][i] = (
                self.cls_01_common["0"][i] / self.cls_01_correct["0"][i]
                if self.cls_01_correct["0"][i] > 0
                else 0
            )
            cls_01_acc["1"][i] = (
                self.cls_01_common["1"][i] / self.cls_01_correct["1"][i]
                if self.cls_01_correct["1"][i] > 0
                else 0
            )
            attr_01_acc["0"][i] = (
                self.attr_01_common["0"][i] / self.attr_01_correct["0"][i]
                if self.attr_01_correct["0"][i] > 0
                else 0
            )
            attr_01_acc["1"][i] = (
                self.attr_01_common["1"][i] / self.attr_01_correct["1"][i]
                if self.attr_01_correct["1"][i] > 0
                else 0
            )

        attr_01_acc["0"][-1] = (
            self.attr_01_common["0"][-1] / self.attr_01_correct["0"][-1]
            if self.attr_01_correct["0"][-1] > 0
            else 0
        )
        attr_01_acc["1"][-1] = (
            self.attr_01_common["1"][-1] / self.attr_01_correct["1"][-1]
            if self.attr_01_correct["1"][-1] > 0
            else 0
        )
        return cls_01_acc, attr_01_acc

    def calculate_auc(self):
        cls_auc = []
        attr_auc = []
        for i in range(self.num_clfs - 1):
            try:
                auc = roc_auc_score(self.cls_labels[i], self.cls_scores[i])
            except ValueError:
                auc = float("nan")  # Handle cases with only one class
            cls_auc.append(auc)

        for i in range(self.num_clfs):
            try:
                auc = roc_auc_score(self.attr_labels[i], self.attr_scores[i])
            except ValueError:
                auc = float("nan")
            attr_auc.append(auc)

        return cls_auc, attr_auc
