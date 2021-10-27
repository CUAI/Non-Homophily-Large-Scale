import sys 
sys.path.insert(1, '../')
import numpy as np
from os import path
import pickle
import requests
import re
import time
import torch
from collections import defaultdict
from collections import deque
import data_utils
import homophily

# This file cleans up the raw data collected from Wikipedia, and prunes nodes that were broken links
# Saves 3 tensors:
# "edges.pt": Stores the graph structure
# "text.pt": Stores the intro text for each Wikipedia page
# "views.pt": Stores the labels for each node, calculated as quintiles based on total views over a period

def clean(text):
    text = text.replace("    ", "")
    text = text.replace("\n", "")
    text = re.sub(r"\s*{.*}\s*", "", text)
    return text


# if loadTensor is true, the edges are loaded up from the edges.pt file
# if loadViews is true, the views are loaded up from the views.pt file

def unpickle(loadTensor=False, loadViews=False):
    text = open("../pagetext.obj", "rb")
    edges = open("../edges.obj", "rb")
    c = open("../checkpoint.obj", "rb")
    visited = pickle.load(c)
    views = pickle.load(c)
    indices = pickle.load(c)
    q = pickle.load(c)
    iters = pickle.load(c)

    indicestoname = {}
    for k in indices:
        indicestoname[indices[k]] = k

    embeddings_dict = defaultdict(
        lambda: np.asarray([0 for i in range(300)], "float32"))
    with open("glove.6B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    intros = []
    ctext = open("../cleanedtext.obj", "wb")
    i = 0
    while (True):
        try:
            index, intro = pickle.load(text)
            i += 1
            print(i)
        except:
            print("end of intro text file")
            print(i, " nodes fully visited")
            break
        if (index < len(intros)):
            print("text already added")
            print(index)
            print(intros[index])

        print(index, len(intros))
        while (index > len(intros)):
            print("missing page")
            intros.append("")
        cleaned = clean(intro)
        intros.append(cleaned)
        pickle.dump((index, cleaned), ctext)

    titles = []
    textvec = []
    for i in range(len(intros)):
        title = indicestoname[i].split()
        print(f"I: {i}") 
        if (len(title) == 0):
            print("0 len title")
            titles.append(np.asarray([0 for i in range(300)], "float32"))
        else:
            titles.append(
                np.mean(np.array([embeddings_dict[word] for word in title]), axis=0))
        intro = intros[i].split()
        if (len(intro) == 0):
            print(title)
            print("0 len intro")
            textvec.append(np.asarray([0 for i in range(300)], "float32"))
        else:
            textvec.append(
                np.mean(np.array([embeddings_dict[word] for word in intro]), axis=0))
    print("Loop finished") 
    titles = np.asarray(titles)
    textvec = np.asarray(textvec)
    print(f"titles: {titles.shape}")
    print(f"textvec: {textvec.shape}")
    merged = np.concatenate((titles, textvec), axis=1)
    print(merged.shape)
    text_tensor = torch.from_numpy(merged)
    torch.save(text_tensor, "text.pt")

    if loadViews:
        views_tensor = torch.load("views.pt")
    else:
        views_np = np.empty(len(intros))
        # views_np = np.empty(len(indices))
        views_np[:] = np.nan
        # views_list = [np.nan for i in range(len(indices))]
        for name in views:
            try:
                # views_list[indices[name]] = views[name]
                if indices[name] >= len(intros):
                    print("uh oh")
                    raise IOError
                    print(name)
                    print(indices[name])
                    continue
                views_np[indices[name]] = views[name]
            except KeyError as e:
                print("key error views ! ! ")
                print(e)
                pass
        print(views_np.shape)

        conv_labels = data_utils.even_quantile_labels(views_np, 5)

        views_tensor = torch.from_numpy(conv_labels)
        torch.save(views_tensor, "views.pt")

    # getting the m x 2 tensor of edges
    if loadTensor:
        edge_tensor = torch.load("edges.pt")
    else:
        edge_list = []
        edge_set = set()
        i = 0
        errs = set()
        while (True):
            try:
                linker, linkee = pickle.load(edges)
                i += 1
            except:
                print("end of edges file")
                print(i, " edges")
                break
            try:
                i_1 = indices[linker]
            except KeyError as e:
                print("key error linker")
                print(e)
                errs.add(e.args[0])
            try:
                i_2 = indices[linkee]
            except KeyError as e:
                print("key error linkee")
                print(e)
                errs.add(e.args[0])
                pass
            try:
                if (indices[linker], indices[linkee]) in edge_set:
                    print(i)
                    print("dupl found")
                    continue
                if (linker not in views or linkee not in views):
                    print(i)
                    print(linker in views)
                    print(linkee in views)
                    print("pruning edge")
                    continue
                edge_list.append([indices[linker], indices[linkee]])
                edge_set.add((indices[linker], indices[linkee]))

                if linker in errs:
                    errs.remove(linker)
                    print(linker, " removed")
                if linkee in errs:
                    errs.remove(linkee)
                    print(linkee, " removed")
            except KeyError as e:
                print("key error ! ! ")
                pass
        print("edges done parsing")
        print()
        edges_np = np.array(edge_list)
        print(edges_np.shape)
        edge_tensor = torch.from_numpy(edges_np)
        torch.save(edge_tensor, "edges.pt")

    A = edge_tensor.T

    print("unique nodes discovered: ", len(indices))

    print(
        f"Edge homophily level: {homophily.edge_homophily_edge_idx(A, views_tensor)}")
    print(
        f"Our measure homophily level: {homophily.our_measure(A, views_tensor)}")

    H = homophily.compat_matrix_edge_idx(A, views_tensor)
    print(H)



unpickle(False, False)
