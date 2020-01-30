import tensorflow as tf
import os
import numpy as np
import subprocess
import pickle
import os
import logging as logger
import pandas as pd
import networkx as nx

EOQ_SYMBOL = 1


def make_attention_mask(X, eoq_symbol=EOQ_SYMBOL):
    def make_mask(last_seen_eoq_pos, query, eoq_symbol):

        if last_seen_eoq_pos == -1:
            return np.zeros((1, len(query)))
        else:
            mask = np.ones([len(query)])
            mask[last_seen_eoq_pos:] = 0.
            mask = np.where(query == eoq_symbol, 0, mask)
            return mask.reshape(1, (len(mask)))

    # eoq_mask = np.where(X == float(EOQ_SYMBOL), float(EOQ_SYMBOL), 0.)
    first_query = True

    for i in range(X.shape[1]):  # loop over batch size --> this gives 80 queries
        query = X[:, i]  # eoq_mask[:, i] #X[:, i]
        # print("query", query)

        first = True
        last_seen_eoq_pos = -1
        # query_masks = []
        for w_pos in range(len(query)):

            if query[w_pos] == float(eoq_symbol):
                last_seen_eoq_pos = w_pos

            if first:
                query_masks = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                first = False
            else:
                new_mask = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                query_masks = np.concatenate((query_masks, new_mask), axis=0)

        global query_masks
        if first_query:
            batch_masks = np.expand_dims(query_masks, axis=2)
            first_query = False
        else:
            batch_masks = np.dstack((batch_masks, query_masks))

    batch_masks = np.transpose(batch_masks, (0, 2, 1))
    # print("shape batch masks", batch_masks.shape)
    # print("batch masks:", batch_masks)
    # print("-------------------------------")
    # print(np.shape())

    return batch_masks  # = attention masks


def x_uniq(X):
    flag = X[0]
    flag = np.array(flag)
    X_new = []
    X_new.append(flag)
    i = 0
    length_list = []
    for sub_x in X[1: ]:
        i += 1
        if not (flag == sub_x).all():
            flag = sub_x
            X_new.append(sub_x)
            length_list.append(i)
            i = 0
    length_list.append(i+1)
    X_new = np.array(X_new)
    return X_new, length_list


def get_gnn_a(X, attention_mask, graph):
    max_length = X.shape[0]
    batch_size = X.shape[1]
    X_expanded = np.tile(np.expand_dims(X, 2), (1, 1, max_length))
    X_expanded = np.transpose(X_expanded, (2, 1, 0))
    X_expanded = X_expanded * attention_mask
    X_expanded = np.transpose(X_expanded, (1, 0, 2))
    X_expanded_reshape = np.reshape(X_expanded, (-1, max_length))
    x_expended_new, length_list = x_uniq(X_expanded_reshape)
    node_list = []
    n_node = []
    items = []
    A = []
    graph_node = list(graph.nodes())
    for x_sub in x_expended_new:
        node = np.unique(x_sub).tolist()
        ner_node_all = []
        for n in node:
            if n in graph_node:
                ner_node = list(graph.adj[n])
                ner_node_all += ner_node
        node += ner_node_all
        node = np.unique(node)
        node_list.append(node)
        n_node.append(len(node))

    max_node = np.max(n_node)
    index = 0
    for x_sub in x_expended_new:
        node = node_list[index]
        items.append(node.tolist() + (max_node - len(node)) * [0])
        ori_node = np.unique(x_sub)
        sub_A = np.zeros((max_node, max_node))
        for n in ori_node:
            if n in graph_node:
                list_j = list(graph.adj[n])
                i = np.where((node == n))[0][0]
                if len(list_j) != 0:
                    for ner_n in list_j:
                        a = graph[n][ner_n]['weight']
                        j = np.where((node == ner_n))[0][0]
                        sub_A[i][j] = a
                        sub_A[j][i] = a

        A.append(sub_A)
        index += 1
    A_new = []
    items_new = []
    for i in range(len(length_list)):
        A_uniq = A[i]
        items_uniq = items[i]
        A_expanded = np.tile(A_uniq, (length_list[i], 1, 1))
        items_expanded = np.tile(items_uniq, (length_list[i], 1))
        if i == 0:
            A_new = A_expanded
            items_new = items_expanded
        else:
            A_new = np.concatenate((A_new, A_expanded), axis=0)
            items_new = np.concatenate((items_new, items_expanded), axis=0)

    A_reshape = np.reshape(A_new, (batch_size, max_length, max_node, max_node))
    items_reshape = np.reshape(items_new, (batch_size, max_length, -1))

    A_reshape = np.transpose(A_reshape, (1, 0, 2, 3))
    items_reshape = np.transpose(items_reshape, (1, 0, 2))

    return A_reshape, items_reshape


def build_aln_graph(ALN_PATH):
    aln_ori = pd.read_csv(ALN_PATH, sep='\t')
    graph = nx.Graph()
    for index, row in aln_ori.iterrows():
        graph.add_edge(row['word_Ni'], row['word_Nj'], weight=row['A(Ni-Nj)'])

    return graph

