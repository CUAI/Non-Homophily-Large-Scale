import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
from os import path as path

DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'

def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat(DATAPATH + 'facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_twitch(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"data/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0] # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label
    
    return A, label, features


def load_pokec():
    pathname = f"{DATAPATH}pokec/"
    node_filename = pathname + 'soc-pokec-profiles.txt'
    with open(node_filename, 'r') as f:
        user_lst = f.readlines()
    label = []
    for user in user_lst:
        gender = user.split('\t')[3]
        gender = int(gender) if gender != 'null' else -1
        label.append(gender)
    label = np.array(label)
    edge_filename = pathname + 'soc-pokec-relationships.txt'
    src = []
    targ = []
    with open(edge_filename, 'r') as f:
        count = 0
        for row in f:
            elts = row.split()
            src.append(int(elts[0]))
            targ.append(int(elts[1]))
            count += 1
            if count % 3000000 == 0:
                print("Loading edges:", count)
    src = np.array(src) - 1
    targ = np.array(targ) - 1
    A = scipy.sparse.csr_matrix((np.ones(len(src)), (src, targ)))
    return A, label

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features