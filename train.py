from model import SWSpeedModel
from Dataloader import Dataset
import numpy as np
from PIL import Image
import os
import sklearn
import random
import glob 
from matplotlib import pyplot as plt

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

# create custom data generator
# data type of images are png
from tensorflow.keras.utils import Sequence
class DataGenerator(Sequence):
    def __init__(self, input_seq, batch_size, img_size, sw_x, img, y, shuffle=True):
        self.batch_size = batch_size
        self.sw_x = sw_x
        self.img = img
        self.y = y
        self.indexes = np.arange(len(self.sw_x))
        self.shuffle = shuffle
        self.img_size = img_size
        
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return len(self.sw_x) // batch_size
    
    def __data_generation__(self, index):
        img_x = []
        for i in range(len(self.img[index])):
            temp = []
            img = Image.open(self.img[index][i][0]).convert("L")
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img) / 255.0
            temp.append(img)
            
            img = Image.open(self.img[index][i][1]).convert("L")
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img) / 255.0
            temp.append(img)
            img_x.append(temp)
            
        img_x = np.array(img_x).reshape((input_seq, 2, self.img_size, self.img_size))
        img_x = np.transpose(img_x, (0, 2, 3, 1))  # shape : (input_seq, img_size, img_size, 2)
        return img_x
        
        
    def __getitem__(self, index): # index : batch number
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size] 
        img_x = []
        sw_x = []
        target = []
        
        for i in range(len(indexes)):
            img_x.append(self.__data_generation__(indexes[i]))
            sw_x.append(self.sw_x[indexes[i]])
            target.append(self.y[indexes[i]])
        return [np.array(sw_x, dtype=np.float16), np.array(img_x, dtype=np.float16)], np.array(target, dtype=np.float16)

# ======== Data preparation ======== #
sw_path = 'your_sw_path'
img_path1 = 'your_sdo_211_path'
img_path2 = 'your_sdo_193_path'

train_sw_x, img_x, y, _ = DataSet(sw_path, img_path1, img_path2, mode='train').get_dataset()
val_sw_x, val_img_x, val_y, _ = DataSet(sw_path, img_path1, img_path2, mode='val').get_dataset()
batch_size = 8

train_gen = DataGenerator(input_seq, batch_size, img_size, train_sw_x, img_x, y)
val_gen = DataGenerator(input_seq, batch_size, img_size, val_sw_x, val_img_x, val_y)

# ======== Model train ========= #
model = SWSpeedModel().build_model()
model.summary()

# make a custom loss function (RMSE)
import tensorflow.keras.backend as K
def rmse(y_true, y_pred):
    loss =  K.sqrt(K.mean(K.square(y_pred*1000 - y_true*1000))) 
    return loss
    
from tensorflow.keras import callbacks
callback = [callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)]

optimizer = Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss=rmse)
train_step = len(train_sw_x) // batch_size
val_step = len(val_sw_x) // batch_size

hist = model.fit_generator(train_gen, validation_data=val_gen, callbacks=[callback], steps_per_epoch=train_step, validation_steps=val_step, epochs=300, workers=200)

# ========= Plot the training process ======== #
fig, loss_ax = plt.subplots()
loss_ax.plot(hist.history['rmse'], 'y', label='train loss')
loss_ax.plot(hist.history['val_rmse'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('rmse')
loss_ax.legend(loc='upper left')
plt.show()

# ========== Save the trained model (architecture & weights)========== #
from tensorflow.keras.callbacks import ModelCheckpoint

model_json = model.to_json()
with open("./model/model.json", "w") as json_file : 
    json_file.write(model_json)
    
model.save_weights("./model/model.h5")
print("Saved model to disk")
