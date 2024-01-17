## Dataloader 
# Input: SDO/AIA images (193 & 211) & OMNIweb SW speed data
# Target: SW speed 

import pandas as pd
import numpy as np
import glob

class Dataset:
    def __init__(self, sw_path, img_path1, img_path2, mode='train'):
        self.sw_path = sw_path
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.mode = mode            # select 'train', or 'val', or 'test'
        self.input_seq_img = 10     # number of input image sequences
        self.input_seq_sw = 20      # number of input sw sequences
        self.output_seq = 12        # number of output sequences
