"""
Adapted from SceneGraphLoc: https://github.com/y9miao/VLSG.
"""

import json
import os
import os.path as osp
import pickle

import numpy as np


def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def load_pkl_data(filename):
    with open(filename, "rb") as handle:
        data_dict = pickle.load(handle)
    return data_dict


def load_json(filename):
    file = open(filename)
    data = json.load(file)
    file.close()
    return data


def idx2name(file_name):
    idx2name = {}
    with open(file_name) as f:
        lines = f.read().splitlines()
        for line in lines:
            split_str = line.split("	")
            idx = split_str[0]
            name = split_str[-1]
            idx2name[int(idx)] = name
    return idx2name


def write_pkl_data(data_dict, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def name2idx(file_name):
    name2idx = {}
    index = 0
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            className = line.split("\n")[0]
            name2idx[className] = index
            index += 1

    return name2idx


def log_softmax_to_probabilities(log_softmax, epsilon=1e-5):
    softmax = np.exp(log_softmax)
    probabilities = softmax / np.sum(softmax)
    assert np.sum(probabilities) >= 1.0 - epsilon and np.sum(probabilities) <= 1.0 + epsilon
    return probabilities


def merge_duplets(duplets):
    merged = []
    for duplet in duplets:
        merged_duplet = None
        for i, m in enumerate(merged):
            if any(id in m for id in duplet):
                if merged_duplet is None:
                    merged_duplet = m
                else:
                    merged_duplet.extend(m)
                    merged.pop(i)
        if merged_duplet is not None:
            merged_duplet.extend(duplet)
        else:
            merged.append(list(duplet))

    merged_set = []
    for merge in merged:
        merged_set.append(sorted(set(merge)))
    return merged_set
