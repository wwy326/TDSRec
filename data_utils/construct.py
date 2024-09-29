from scipy.sparse import csr_matrix
import numpy as np
import pickle
import csv

import copy
import os

def construct_graphs(seqs, num_items, distance, prefix):
    user = list()
    r, c, d = list(), list(), list()
    for i, seq in enumerate(seqs):
        print(f"Processing {i}/{len(seqs)} (>﹏<)    ", end='\r')
        for dist in range(1, distance + 1):
            if dist >= len(seq): break;
            r += copy.deepcopy(seq[+dist:])
            c += copy.deepcopy(seq[:-dist])
            r += copy.deepcopy(seq[:-dist])
            c += copy.deepcopy(seq[+dist:])
    d = np.ones_like(r)
    iigraph = csr_matrix((d, (r, c)), shape=(num_items, num_items))
    print('Constructed i-i graph, density=%.6f' % (len(d) / (num_items ** 2)))
    with open(prefix + 'trn', 'wb') as fs:
        pickle.dump(iigraph, fs)

if __name__ == '__main__':

    # dataset = input('Choose a dataset: ')
    dataset = 'cd'
    prefix = '../datasets/sequential/' + dataset + '/' + 'cd_item.csv'

    # distance  = int(input('Max distance of edge: '))
    distance = 472265

    data=[]
    with open(prefix, 'r') as fs:
        csv_reader = csv.reader(fs)
        for row in csv_reader:
            data.append(row)

    data = [[int(x) for x in string_list[0].split()] for string_list in data]

    if dataset == ('cd'):
        num_items =  35118
    elif dataset == ('Games'):
        num_items =  23715
    elif dataset == ('Beauty'):
        num_items =  57289

    construct_graphs(data, num_items, distance, prefix)
