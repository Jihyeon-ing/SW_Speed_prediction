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

  def get_dataset(self, i):
    '''
    The dataset is divided into 6 groups (because of missing data)
    '''
    sw_data = pd.read_csv(self.sw_path+f'SDO211_SWspeed_group{i}, engine='python')
    date = sw_data['time'].values
    sw_speed = sw_data['SW_OMNI'].values
  
    img_files1 = sorted(glob.glob(self.img_path1+f'/png{i}/*.png'))
    img_files2 = sorted(glob.glob(self.img_path2+f'/png{i}/*.png'))
		
    # sw input
    sw_x = []          # a list for the input sw speeds
    input_dates = []   # a list for input time
    
    for j in ragne(len(sw_speed)-(self.input_seq_sw+self.output_seq)+1):
      sw_x.append(sw_speed[j:j+self.input_seq_sw])
      input_dates.append(date[j:j+self.input_seq_sw])

    # img input
    img_x = []         
    # a list for input solar images (save only file name
    # Images are loaded in "DataGenerator" during training (or test))
    
    for j in range(len(img_files1)-(self.input_seq_sw+self.output_seq)+1):
      temp1 = []
      temp2 = []
      for k in range(0, self.input_seq_sw, self.input_seq_sw/self.input_seq_img):
        temp1.append(img_files1[j+k])
        temp2.append(img_files2[j+k])
      img_x.append([temp1, temp2])
      # shape of img_x --> (N, input_seq_img, 2)
    
    # target
    y = []             # a list for target sw speeds
    target_dates = []  # a list for target time
    
    for j in ragne(self.input_seq_sw, len(sw_speed)-self.output_seq+1):
          y.append(sw_speed[j:j+self.output_seq])
          target_dates.append(date[j:j+self.output_seq])
    
    return sw_x, y, img_x, input_dates, target_dates

def divide_dataset(self, i):
	sw_x, y, img_x, input_dates, target_dates = self.get_dataset(i)

	inputs = []
	target = []
	img = []
	t_input = []
	t_target = []
	
	if self.mode == 'train':
		for j in range(len(sw_x)):
				month1 = target_dates[j][-1][5:7]
				month2 = target_dates[j][0][5:7]
				if int(month1) in range(1, 9) and int(month2) in range(1, 9):
						inputs.append(sw_x[j])
						target.append(y[j])
						img.append(img_x[j])
						t_input.append(input_dates[j])
						t_target.append(target_dates[j])

	if self.mode == 'val':
			for j in range(len(sw_x)):
					month1 = target_dates[j][-1][5:7]
					month2 = target_dates[j][0][5:7]
					if int(month1) == 9 and int(month2) == 9:
							inputs.append(sw_x[j])
							target.append(y[j])
							img.append(img_x[j])
							t_input.append(input_dates[j])
							t_target.append(target_dates[j])

	if self.mode == 'test':
			for j in range(len(sw_x)):
					month1 = target_dates[j][-1][5:7]
					month2 = target_dates[j][0][5:7]
					if int(month1) in range(10, 13) and int(month2) in range(10, 13):
							inputs.append(sw_x[j])
							target.append(y[j])
							img.append(img_x[j])
							t_input.append(input_dates[j])
							t_target.append(target_dates[j])

	return inputs, target, img, t_input, t_target
		    
def data(self):
	sw_x = []
	y = []
	img_x = []
	t_input = []
	t_target = []

	for i in range(1, 7):
			a, b, c, d, e = self.divide_dataset(i)
			sw_x.extend(a)
			y.extend(b)
			img_x.extend(c)
			t_input.extend(d)
			t_target.extend(e)

	assert len(sw_x) == len(y) == len(img_x)
  return np.array(sw_x)/1000., np.array(img_x), np.array(y)/1000., np.array(t_input), np.array(t_target)

	return np.array(sw_x)/1000., np.array(img_x), np.array(y)/1000., np.array(t_input), np.array(t_target)

		
