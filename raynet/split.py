import json #create the json
import shutil #copy images to train, test and valid dirs
import os #files and dirs manipulation
import math #split calculate

#path configuration
parent_dir = ‘/content/gdrive/My Drive/assignment_dataset’#get category folder list
os.chdir(parent_dir)
category_list = list(filter(lambda x: os.path.isdir(x), os.listdir()))
for category in category_list:
  print(category)

  #create training,validation,testing directories
data_set_dirs= ['train','valid','test']
for dsdirs in data_set_dirs:
  path = parent_dir + '/'+ dsdirs
  os.mkdir( path,755 )	

  #define proportion of data
train_prop = 0.6
valid_prop = test_prop = (1-train_prop)/2#function to split data of each category into trainning, validation and testing set
def create_dataset():
  for ii,cat in enumerate(category_list):    
    src_path = parent_dir + '/' + cat
    dest_dir1 = parent_dir+'/train/'+str(ii)
    dest_dir2 = parent_dir+'/valid/'+str(ii)
    dest_dir3 = parent_dir+'/test/'+str(ii)
    
    dest_dirs_list = [dest_dir1,dest_dir2,dest_dir3]
    for dirs in dest_dirs_list:
      os.mkdir(dirs,755 )
    
    #get files' names list from respective directories
    os.chdir(src_path)
    files = [f for f in os.listdir() if os.path.isfile(f)]
    
    #get training, testing and validation files count
    train_count = math.ceil(train_prop*len(files))
    valid_count = int((len(files)-train_count)/2)
    test_count = valid_count
    
    #get files to segragate for train,test and validation data set
    train_data_list = files[0: train_count]
    valid_data_list = files[train_count+1:train_count+1+valid_count]  
    test_data_list =  files[train_count+valid_count:]
       
  
    for train_data in train_data_list:
      train_path = src_path + '/' + train_data
      shutil.copy(train_path,dest_dir1)
    
    for valid_data in valid_data_list:
      valid_path = src_path + '/' + valid_data
      shutil.copy(valid_path,dest_dir2)
    
    for test_data in test_data_list:
      test_path = src_path + '/' + test_data
      shutil.copy(test_path,dest_dir3)

#save category data as dictionary in a json file
cat_data = {}
for ix,cat in enumerate(category_list):
  cat_data[ix] = cat
with open('/content/gdrive/My Drive/assignment_dataset/cat_data.json', 'w') as outfile:  
    json.dump(cat_data , outfile)