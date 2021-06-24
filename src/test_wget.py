# -*- coding: utf-8 -*-
# @Time    : 6/23/21 1:25 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : test_wget.py

import wget

import wget

# DATA_URL = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
#
# wget.download(DATA_URL, out='../pretrained_models/ast_audioset.pth')

import zipfile
import os
os.mkdir('../esc50')
with zipfile.ZipFile('../master.zip', 'r') as zip_ref:
    zip_ref.extractall('../esc50')