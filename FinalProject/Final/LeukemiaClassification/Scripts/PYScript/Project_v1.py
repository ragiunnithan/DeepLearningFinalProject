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
from tensorflow.keras.preprocessing import image

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

val_data = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data"
val_labels = pd.read_csv("/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv")




#%%
#########################Reading the training dataset######################
##Read the cells with ALL - Acute Lymphoblastic Leukemia (ALL)

can_image_fnames = os.listdir(fold0_all)

##Plotting a single image
cancer_img = imageio.imread(os.path.join(fold0_all,
                                         can_image_fnames[5]))
plt.imshow(cancer_img)
plt.show()



#blood cell without cancer - normal cells
nrml_image_fnames = os.listdir(fold0_nrml)
nrml_img = imageio.imread(os.path.join(fold0_nrml,
                                         nrml_image_fnames[5]))

plt.imshow(nrml_img)
plt.show()

#%%

##Get the shape of the image 
nrml_img.shape

#The image is 450 * 450 poxel with colors

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

plt.pie([len(train_df[train_df["y_col"]=="ALL"]),len(train_df[train_df["y_col"]=="HEM"])],
        labels=["ALL","Normal"],autopct='%.f'
       )
plt.title('Pie Chart for percentage of each cell type')
plt.gca()
plt.show()


#%%
select_normal = np.random.choice(normal_lst, 3, replace = False)
select_all = np.random.choice(cancer_lst, 3, replace = False)


fig = plt.figure(figsize = (8,6))

for i in range(6):
    if i < 3:
        fp = select_normal[i]
        label = 'Normal'
    else:
        fp = select_all[i-3]
        label = 'ALL'
    ax = fig.add_subplot(2, 3, i+1)
    fn = image.load_img(fp, target_size = (100,100),
                        color_mode='rgb')
    plt.imshow(fn, cmap='Greys_r')
    plt.title(label)
    plt.axis('off')
plt.show()




plt.bar(['Normal', 'ALL'], [len(normal_lst), len(cancer_lst)])
plt.title('Original Class Imbalance')
plt.show()


#%%
def img2np(fn_list, size = (32, 32)):
   
    i = 0
    for fp in fn_list:
        
        current_image = image.load_img(fp, 
                                       target_size = size, 
                                       color_mode = 'grayscale')
        
        img_ts = image.img_to_array(current_image)
        img_ts = [img_ts.ravel()]
             
        
        
        
        if i == 0:
            full_mat = img_ts
            
        else: 
            full_mat = np.concatenate((full_mat, img_ts))    
        i = i + 1    
    return full_mat



normal_npArray = img2np(normal_lst)
cancer_npArray = img2np(cancer_lst)


##Let us select 25 images to make the comparison clear









