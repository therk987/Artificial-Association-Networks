import pickle
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch
import random
import numpy as np
from representation.utils import A2Edge

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def ALGORITHM_AST_NEUROTREE_DATA(random_state = 0):

    root = os.path.dirname(os.path.realpath(__file__)) + '/codeclone/algo/result/algo_ast_neurotree.pkl'
    with open(root, 'rb') as f:
        data = pickle.load(f)

    labeldict = defaultdict(lambda: None)

    X = []
    Y = []
    for key, value1, value2 in data:
        labelname = key.split('/')[-2]
        if labeldict[labelname]:
            label = labeldict[labelname]
        else:
            label = len(labeldict.keys()) - 1
            labeldict[labelname] = label

        X.append((value1, value2, key))
        Y.append(label)
        # print(key.split('/')[-2])
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, )

    train_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_train, Y_train):
        x1, x2, f = xf
        train_dataset['x'].append((x1, x2))
        train_dataset['y'].append(y)
        train_dataset['f'].append(f)

    test_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_test, Y_test):
        x1, x2, f = xf
        test_dataset['x'].append((x1, x2))
        test_dataset['y'].append(y)
        test_dataset['f'].append(f)

    print('LOAD : AST NEUROTREE ALGORITHMS')
    print('train : ', len(train_dataset['x']))
    print('test : ', len(test_dataset['x']))
    neurotrees_dict = {}
    neurotrees_dict['train'] = train_dataset
    neurotrees_dict['test'] = test_dataset
    return neurotrees_dict


def ALGORITHM_AST_NEUROTREE_FUNC_CALL_DATA(random_state = 0):

    root = os.path.dirname(os.path.realpath(__file__)) + '/codeclone/algo/result/algo_ast_neurotree_func_call-v2.pkl'
    with open(root, 'rb') as f:
        data = pickle.load(f)

    labeldict = defaultdict(lambda: None)

    X = []
    Y = []
    for key, value1, value2 in data:
        labelname = key.split('/')[-2]
        if labeldict[labelname]:
            label = labeldict[labelname]
        else:
            label = len(labeldict.keys()) - 1
            labeldict[labelname] = label

        X.append((value1, value2, key))
        Y.append(label)
        # print(key.split('/')[-2])
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=test_size, random_state=random_state, stratify=Y_train)

    train_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_train, Y_train):
        x1, x2, f = xf
        train_dataset['x'].append((x1, x2))
        train_dataset['y'].append(y)
        train_dataset['f'].append(f)

    valid_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_valid, Y_valid):
        x1, x2, f = xf
        valid_dataset['x'].append((x1, x2))
        valid_dataset['y'].append(y)
        valid_dataset['f'].append(f)



    test_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_test, Y_test):
        x1, x2, f = xf
        test_dataset['x'].append((x1, x2))
        test_dataset['y'].append(y)
        test_dataset['f'].append(f)

    print('LOAD : AST NEUROTREE FUNC CALL ALGORITHMS')
    print('train : ', len(train_dataset['x']))
    print('valid : ', len(valid_dataset['x']))
    print('test : ', len(test_dataset['x']))
    neurotrees_dict = {}
    neurotrees_dict['train'] = train_dataset
    neurotrees_dict['valid'] = valid_dataset
    neurotrees_dict['test'] = test_dataset
    return neurotrees_dict


def SORT_AST_NEUROTREE_CONTROL_FLOW_GRAPH_DATA(random_state = SEED):

    root = os.path.dirname(os.path.realpath(__file__)) + '/codeclone/sort/result/sort_ast_neurotree_cfg.pkl'
    with open(root, 'rb') as f:
        data = pickle.load(f)

    labeldict = defaultdict(lambda: None)

    X = []
    Y = []
    for key, value1, value2 in data:
        labelname = key.split('/')[-2]
        if labeldict[labelname]:
            label = labeldict[labelname]
        else:
            label = len(labeldict.keys()) - 1
            labeldict[labelname] = label

        X.append((value1, value2, key))
        Y.append(label)
        # print(key.split('/')[-2])
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=test_size, random_state=random_state, stratify=Y_train)

    train_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_train, Y_train):
        x1, x2, f = xf
        x2.A_c = A2Edge(x2.A_c)
        train_dataset['x'].append((x1, x2))
        train_dataset['y'].append(y)
        train_dataset['f'].append(f)

    valid_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_valid, Y_valid):
        x1, x2, f = xf
        x2.A_c = A2Edge(x2.A_c)
        valid_dataset['x'].append((x1, x2))
        valid_dataset['y'].append(y)
        valid_dataset['f'].append(f)



    test_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_test, Y_test):
        x1, x2, f = xf
        x2.A_c = A2Edge(x2.A_c)
        test_dataset['x'].append((x1, x2))
        test_dataset['y'].append(y)
        test_dataset['f'].append(f)

    print('LOAD : AST NEUROTREE FUNC CALL ALGORITHMS')
    print('train : ', len(train_dataset['x']))
    print('valid : ', len(valid_dataset['x']))
    print('test : ', len(test_dataset['x']))
    neurotrees_dict = {}
    neurotrees_dict['train'] = train_dataset
    neurotrees_dict['valid'] = valid_dataset
    neurotrees_dict['test'] = test_dataset
    return neurotrees_dict


def ALGORITHM_NEUROTREE_DATA(random_state = 0):
    root = os.path.dirname(os.path.realpath(__file__)) + '/codeclone/algo/result/algo_neurotree.pkl'
    with open(root, 'rb') as f:
        data = pickle.load(f)

    labeldict = defaultdict(lambda: None)

    X = []
    Y = []
    for key, value in data:
        labelname = key.split('/')[-2]
        if labeldict[labelname]:
            label = labeldict[labelname]
        else:
            label = len(labeldict.keys()) - 1
            labeldict[labelname] = label

        X.append((value, key))
        Y.append(label)
        # print(key.split('/')[-2])
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

    train_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_train, Y_train):
        x, f = xf
        train_dataset['x'].append(x)
        train_dataset['y'].append(y)
        train_dataset['f'].append(f)

    test_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_test, Y_test):
        x, f = xf
        test_dataset['x'].append(x)
        test_dataset['y'].append(y)
        test_dataset['f'].append(f)

    print('LOAD : ALGORITHMS')
    print('train : ', len(train_dataset['x']))
    print('test : ', len(test_dataset['x']))
    neurotrees_dict = {}
    neurotrees_dict['train'] = train_dataset
    neurotrees_dict['test'] = test_dataset
    return neurotrees_dict


def ALGORITHM_AST_DATA(random_state = 0):
    root = os.path.dirname(os.path.realpath(__file__)) + '/codeclone/algo/result/algo_ast.pkl'
    with open(root, 'rb') as f:
        data = pickle.load(f)
    labeldict = defaultdict(lambda: None)

    X = []
    Y = []
    for key, value in data:
        labelname = key.split('/')[-2]
        if labeldict[labelname]:
            label = labeldict[labelname]
        else:
            label = len(labeldict.keys()) - 1
            labeldict[labelname] = label

        X.append((value, key))
        Y.append(label)
        # print(key.split('/')[-2])
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)

    train_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_train, Y_train):
        x, f = xf
        train_dataset['x'].append(x)
        train_dataset['y'].append(y)
        train_dataset['f'].append(f)

    test_dataset = {'x': [], 'y': [], 'f': []}
    for xf, y in zip(X_test, Y_test):
        x, f = xf
        test_dataset['x'].append(x)
        test_dataset['y'].append(y)
        test_dataset['f'].append(f)

    print('LOAD : ALGORITHMS')
    print('train : ', len(train_dataset['x']))
    print('test : ', len(test_dataset['x']))
    neurotrees_dict = {}
    neurotrees_dict['train'] = train_dataset
    neurotrees_dict['test'] = test_dataset
    return neurotrees_dict



