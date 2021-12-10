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
plt.title('Cancer')
#plt.show()
plt.savefig('/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/Cancer.png')


#blood cell without cancer - normal cells
nrml_image_fnames = os.listdir(fold0_nrml)
nrml_img = imageio.imread(os.path.join(fold0_nrml,
                                         nrml_image_fnames[1]))

plt.imshow(nrml_img)
#plt.show()
plt.title('Normal')
plt.savefig('/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/Normal.png')

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
plt.title('% of Normal and Cancer Cells')
plt.gca()
#plt.show()
plt.savefig('/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/Pie_Chart.png')

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
#plt.show()

plt.savefig('/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/Normal_Cancer.png')
#%%

plt.bar(['Normal', 'ALL'], [len(normal_lst), len(cancer_lst)])
plt.title('Bar chart showing the number of images in each cell type')
#plt.show()

plt.savefig('/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/Bar_Chart.png')
#%%

#function for processing images to numpy array for creating mean
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

#%%
##Let us select 50 images to make the comparison clear


nrml_images_sub = normal_npArray[:50]
cnr_images_sub = cancer_npArray[:50]


def plt_mean(mat , title, size = (32, 32)):
    
    mean_img = np.mean(mat , axis = 0)
    mean_img = mean_img.reshape(size)
    plt.imshow(mean_img, vmin=0, vmax=255)
    plt.title(f'Average {title}')
    plt.axis('off')
    #plt.show()
    fname = '/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Plots/' + title + '_mean.png'
    plt.savefig(fname)
    return mean_img

nrml_mean = plt_mean(nrml_images_sub , "Normal")
cnr_mean = plt_mean(cnr_images_sub, "Cancer - ALL")
 

#%% 
#Reading the validation data
validation_list = get_path_image(val_data)

##Convert the list to a dictionary. The labels are stored in the val_labels
#3So we create a dictionary with x as the file name and y as the labels
##The labels are having 0's and 1's.
##0 means normal and 1 means cancer - ALL

validation_dict = {"x_col":validation_list,
                   "y_col":val_labels["labels"]}

validation_df = pd.DataFrame(validation_dict)

validation_df["y_col"].replace(to_replace = [1,0], value = ["ALL","HEM"], inplace = True)

#%%

#Reading the test dataset     

test_data = "/Users/gunnikrishnan/Documents/Ragi/HoodCollege/FirstSem/DeepLearning/Scripts/Project/Data/C-NMC_Leukemia/testing_data/C-NMC_test_final_phase_data/"
test_list = get_path_image(test_data)

test_dict = {"x_col":test_list}

test_df = pd.DataFrame(test_dict)

#%%

##With keras, image preprocessing has become much easier. Instead of reading subfolders for all files, load pictures and converting it to numpy arrays
#Keras provides API calls. flow_from_dataframe allows us to input a pandas dataframe which contains the filenames, with or without extension, as one column and
# and a column which has the class names and directly read the images from the directory with their respective class names mapped.
train_datagen = ImageDataGenerator(
        rescale=1./255 #pixel values are 255 maximum
         )


test_datagen = ImageDataGenerator(
        rescale=1./255 )

train_generator = train_datagen.flow_from_dataframe(
                  train_df,
                  x_col = "x_col",
                  y_col = "y_col",
                  target_size = (256, 256),
                 
                  #batch_size = 32,
                  color_mode = "rgb",
                  shuffle = True,
                  class_mode = "binary")

validation_generator = train_datagen.flow_from_dataframe(
                  validation_df,
                  x_col = "x_col",
                  y_col = "y_col",
                  target_size = (256, 256),                  
                  #batch_size = 32,
                  color_mode = "rgb",
                  shuffle = True,
                  class_mode = "binary")

test_generator = test_datagen.flow_from_dataframe(
                  test_df,
                  x_col = "x_col",
                  target_size = (256, 256),
                  color_mode = "rgb",
                  class_mode = None,
                  shuffle = False)


#%%%

#A Model with one convolution and one dense
#Convolutional layer uses fewer parameters by forcing input values to share the parameters.
##Dense layer uses a linear operation, meaning, every output is formed by the function based on every input.
##In other words, every input is forced into the function, and then the "Neural Network" learns it's relation to the output. 
#There will be n*m connections, where n denotes the number of inputs and m denotes the number of outputs.

##The output of the convolutional layer is formed by just a small size of inputs which depends on the filter's size and the weights are shared 
##for all pixels. The output is constructed by using the same co-efficient for all the pixels by using the neighbouring pixels as inputs.



model1 = models.Sequential()
model1.add(layers.Conv2D(64, 3, activation = 'relu', input_shape = (256, 256, 3)))

model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(512, activation = 'relu'))
model1.add(layers.Dense(1, activation='sigmoid'))


#%%
#cell 12
# compiling models
model1.compile(loss='binary_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy', 'Recall'])

#%%
#cell 13
start = timer()

history = model1.fit(train_generator , 
                    epochs=5, 
                    validation_data=validation_generator, 
                    workers = 7
                   )

end = timer()
elapsed = end - start
print('Total Time Elapsed: ', int(elapsed//60), ' minutes ', (round(elapsed%60)), ' seconds')
#%%
scores = model1.evaluate(test_generator, verbose=1)












