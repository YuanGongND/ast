# -*- coding: utf-8 -*-
# @Time    : 11/17/20 3:22 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_weight_file.py

# gen weight = 1 / num_sample in the dataset

import argparse
import json
import numpy as np
import sys
sys.path.append('/data/sls/scratch/yuangong/aed-pc/src/dataloaders')
from index_lookup import make_index_dict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str, default='/data/sls/scratch/yuangong/audioset/datafiles/speechcommand_train_data.json', help="the root path of data json file")

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path

    index_dict = make_index_dict('/data/sls/scratch/yuangong/ast/egs/speechcommands/speechcommands_class_labels_indices.csv')
    label_count = np.zeros(35)

    with open(data_path, 'r', encoding='utf8')as fp:
        data = json.load(fp)
        data = data['data']

    for sample in data:
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    # the reason not using 1 is to avoid underflow for majority classes, add small value to avoid underflow
    label_weight = 1000.0 / (label_count)
    sample_weight = np.zeros(len(data))

    for i, sample in enumerate(data):
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            sample_weight[i] += label_weight[label_idx]
    np.savetxt(data_path[:-5]+'_weight.csv', sample_weight, delimiter=',')



