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

mdl_list_m = ['/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4495.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4483.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.4475.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_12_12_0.4467.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_14_14_0.4431.pth',
'/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_16_16_0.4422.pth']

mdl_list_m = ['/data/sls/scratch/yuangong/ast/pretrained_models/ensemble/audioset_10_10_0.439.pth']

for mdl in mdl_list_m:
    print(mdl)
    sd = torch.load(mdl, map_location=device)
    fstride, tstride = int(mdl.split('/')[-1].split('_')[1]), int(mdl.split('/')[-1].split('_')[2].split('.')[0])
    audio_model = models.ASTModel(fstride=fstride, tstride=tstride, imagenet_pretrain=False)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)
    torch.save(audio_model.state_dict(), '/data/sls/scratch/yuangong/ast/pretrained_models/legal/'+mdl.split('/')[-1])