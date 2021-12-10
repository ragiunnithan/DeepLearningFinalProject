# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np 
from timeit import default_timer as timer
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
#%%

##Saving all the file names

#ALL is the data with Acute Lymphoblastic Leukemia (ALL)

fold0_all = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_0/all"
fold1_all = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_1/all"
fold2_all = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_2/all"

#hem is the normal one
fold0_nrml = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_0/hem"
fold1_nrml = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_1/hem"
fold2_nrml = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/training_data/fold_2/hem"

test_data = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data"
test_labels = pd.read_csv("/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv")


def get_path_image(folder):
    image_paths = []
    image_fnames = os.listdir(folder) 
    for img_id in range(len(image_fnames)):
        img = os.path.join(folder,image_fnames[img_id])
        image_paths.append(img)
    
    return image_paths

#image absolute paths for cancer cells and normal cells
cancer_lst = []

for i in [fold0_all,fold1_all,fold2_all]:
    paths = get_path_image(i)
    cancer_lst.extend(paths)
    
    
print(len(cancer_lst))

normal_lst = []
for i in [fold0_nrml,fold1_nrml,fold2_nrml]:
    paths = get_path_image(i)
    normal_lst.extend(paths)
   
   
print(len(normal_lst))


cancer_dict = {"x_col":cancer_lst,
          "y_col":[np.nan for x in range(len(cancer_lst))]}


cancer_dict["y_col"] = "ALL"

normal_dict = {"x_col":normal_lst,
          "y_col":[np.nan for x in range(len(normal_lst))]}


normal_dict["y_col"] = "HEM"

cancer_df = pd.DataFrame(cancer_dict)
normal_df = pd.DataFrame(normal_dict)

train_df = cancer_df.append(normal_df, ignore_index=True)

all_len = len(cancer_df)
norm_len = len(normal_df)

nrm_size = int((norm_len/100) * 25)
all_size = int((all_len/100) * 25)
validation_df = pd.DataFrame()
validation_df = validation_df.append(normal_df[0:nrm_size])
validation_df = validation_df.append(cancer_df[0:all_size])    
validation_df["y_col"].replace(to_replace=[1,0],value=["ALL","HEM"],inplace=True)

train_df = pd.DataFrame()
train_df = train_df.append(normal_df[nrm_size : norm_len])
train_df = train_df.append(cancer_df[all_size : all_len])   
train_df["y_col"].replace(to_replace=[1,0],value=["ALL","HEM"],inplace=True)

test_list = get_path_image(test_data)


##Convert the list to a dictionary. The labels are stored in the val_labels
#3So we create a dictionary with x as the file name and y as the labels
##The labels are having 0's and 1's.
##0 means normal and 1 means cancer - ALL

test_dict = {"x_col":test_list,
                   "y_col":test_labels["labels"]}

test_df = pd.DataFrame(test_dict)

test_df["y_col"].replace(to_replace = [1,0], value = ["ALL","HEM"], inplace = True)















