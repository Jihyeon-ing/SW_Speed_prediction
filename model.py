import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

class SWSpeedModel:
  def __init__(self, img_size=64, input_seq=10, n_ch=96):
    self.img_size = img_size
    self.input_seq = input_seq
    self.n_ch = n_ch
    
  # model based on inception block
  def inception(self, x, ch_num):
      pre_layer = x
  
      conv1 = Conv3D(ch_num, kernel_size=(1, 1, 1), padding='same', activation='relu')(pre_layer)
      conv2 = Conv3D(ch_num, kernel_size=(1, 1, 1), padding='same', activation='relu')(pre_layer)
      conv2 = Conv3D(ch_num, kernel_size=(1, 3, 3), padding='same', activation='relu')(conv2)
      conv3 = Conv3D(ch_num, kernel_size=(1, 1, 1), padding='same', activation='relu')(pre_layer)
      conv3 = Conv3D(ch_num, kernel_size=(1, 5, 5), padding='same', activation='relu')(conv3)
      max_pool = MaxPooling3D(pool_size=(1, 3, 3), strides=1, padding='same')(pre_layer)
      max_pool = Conv3D(ch_num, kernel_size=(1, 1, 1), padding='same', activation='relu')(max_pool)
      #output shape = (None, w, h, ch)
  
      concat = concatenate([conv1, conv2, conv3, max_pool], axis=-1)
      return concat

  def build_model(self):
    img_inputs = Input(shape=(self.input_seq, self.img_size, self.img_size, 2))
    img_x = Conv3D(self.n_ch, kernel_size=(1,7,7), padding='same', activation='relu')(img_inputs)
    img_x = MaxPooling3D(pool_size=(1, 3,3), strides=(1,2,2), padding='same')(img_x)
    for _ in range(3):
        img_x = inception(img_x, self.n_ch)
        img_x = MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same')(img_x)
    
    img_x = Reshape((self.input_seq, -1))(img_x)
    img_x = LSTM(512)(img_x)
    img_x = Activation('relu')(img_x)
    
    sw_inputs = Input(shape=(self.input_seq*2, ))
    sw_x = Dense(256, activation='selu')(sw_inputs)
    sw_x = Dense(256, activation='selu')(sw_x)
    
    x = Concatenate(axis=-1)([img_x, sw_x])
    x = Dense(128, activation='relu')(x)
    x = Dense(12)(x)
    
    model = Model([sw_inputs, img_inputs], x)
    return model


