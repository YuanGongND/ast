# -*- coding: utf-8 -*-
# @Time    : 10/19/20 5:15 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_esc50.py

import numpy as np
import json
import os

# label = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_label.csv', delimiter=',', dtype='str')
# f = open("/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices.csv", "w")
# f.write("index,mid,display_name\n")
#
# label_set = []
# idx = 0
# for j in range(0, 5):
#     for i in range(0, 10):
#         cur_label = label[i][j]
#         cur_label = cur_label.split(' ')
#         cur_label = "_".join(cur_label)
#         cur_label = cur_label.lower()
#         label_set.append(cur_label)
#         f.write(str(idx)+',/m/07rwj'+str(idx).zfill(2)+',\"'+cur_label+'\"\n')
#         idx += 1
# f.close()
#
label_set = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc_class_labels_indices_pro.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

for fold in [1,2,3,4,5]:
    base_path = "/data/sls/scratch/yuangong/audioset/ESC-50-master/audio/"
    meta = np.loadtxt('/data/sls/scratch/yuangong/aed-pc/src/utilities/esc50_meta.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for i in range(0, len(meta)):
        cur_label = label_map[meta[i][3]]
        cur_path = meta[i][0]
        cur_fold = int(meta[i][1])
        cur_dict = {"video_id": 'dummy', "wav": base_path + cur_path, "image": 'empty', "labels": '/m/07rwj'+cur_label.zfill(2)}
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)

    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open('/data/sls/scratch/yuangong/audioset/datafiles/esc_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('/data/sls/scratch/yuangong/audioset/datafiles/esc_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)