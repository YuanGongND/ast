# -*- coding: utf-8 -*-
# @Time    : 6/21/21 6:12 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : convert_model.py

# convert hard coded models to cleaned up models

import os, sys
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import models
import torch
import dataloader
import numpy as np
import argparse
from traintest import train, validate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#sd = torch.load('/Users/yuan/Documents/ast/pretrained_models/audio_model_wa.pth', map_location=device)
sd = torch.load('/data/sls/scratch/yuangong/ast/pretrained_models/audio_model_wa.pth', map_location=device)
audio_model = models.ASTModel(pretrain=False)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)

#
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_path", type=str, default='/data/sls/scratch/yuangong/audioset/datafiles/speechcommand_train_data.json', help="the root path of data json file")
args = parser.parse_args()

args.dataset='audioset'
args.data_eval='/data/sls/scratch/yuangong/audioset/datafiles/eval_data.json'
args.label_csv='/data/sls/scratch/yuangong/ast/egs/audioset/class_labels_indices.csv'
args.exp_dir='./tmp/'
args.loss_fn = torch.nn.BCEWithLogitsLoss()
norm_stats = {'audioset': [-4.2677393, 4.5689974], 'esc50': [-6.6268077, 5.358466],
              'speechcommands': [-6.845978, 5.5654526]}
target_length = {'audioset': 1024, 'esc50': 512, 'speechcommands': 128}
# if add noise for data augmentation, only use for speech commands
noise = {'audioset': False, 'esc50': False, 'speechcommands': True}

val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False}
eval_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=24, shuffle=False, num_workers=32, pin_memory=True)
stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
mAP = np.mean([stat['AP'] for stat in stats])
print('---------------evaluate on the test set---------------')
print("mAP: {:.6f}".format(mAP))