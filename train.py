from model import SWSpeedModel
from Dataloader import Dataset
import numpy as np
from PIL import Image
import os
import sklearn
import random
import glob 

# seed 
def my_seed_everywhere(seed):
    random.seed(seed) # random
    np.random.seed(seed) # np
    os.environ["PYTHONHASHSEED"] = str(seed) # os
    tf.random.set_seed(seed) # tensorflow
my_seed = 777
my_seed_everywhere(my_seed)

# gpu 
gpu_id = 0
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

model = SWSpeedModel().build_model()
