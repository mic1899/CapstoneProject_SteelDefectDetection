import random
import os
import time
import pandas as pd

import shutil
import cv2

from sklearn.model_selection import train_test_split

# self-written scripts
import sys
sys.path.insert(0, 'Python_Scripts')

import mask_conversion


def create_train_test_dfs(df, seed):
    X = df.copy()
    y = X.pop('ClassId')

    # Split into train and test set 
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=seed)
    
    return X_train.join(y_train), X_test.join(y_test)



# Create a temp path for the train & test split
def create_train_test_folders(subfolder, class_id):
    print('preparing folders...')
    path = os.getcwd()
    #
    ## You might need to adjust the path to your local environment
    temp_path = path + "/data/" + subfolder
    path_suffix = 'c' + str(class_id)
    
    # make base folder structure
    try:
        os.mkdir(temp_path)
        
        os.mkdir(temp_path + '/train')
        os.mkdir(temp_path + '/train_aug')
        
        os.mkdir(temp_path + '/train_mask')
        os.mkdir(temp_path + '/train_mask_aug')
        
        os.mkdir(temp_path + '/test')
        os.mkdir(temp_path + '/test_mask')
        print('base folder structure created')
        
    except OSError:
        print('base folder structure exists')
    
    # make class specific sub-folder structure
    try:
        os.mkdir(temp_path + '/train/' + path_suffix)
        os.mkdir(temp_path + '/train_aug/' + path_suffix)
        os.mkdir(temp_path + '/train_mask/'+ path_suffix)
        os.mkdir(temp_path + '/train_mask_aug/'+ path_suffix)
        print(f'sub-folder structure for ClassId {class_id} created')
        
    except OSError:
        print(f'sub-folder structure for ClassId {class_id} already exists')
    
    # make class specific sub-folder structure for test
    try:
        os.mkdir(temp_path + '/test/' + path_suffix)
        os.mkdir(temp_path + '/test_mask/' + path_suffix)
        print(f'sub-foder structure for testing for ClassId {class_id} created')
        
    except OSError:
        print ("Directories already exist")
    else:
        print ("Successfully created the directories")
    print()



# Copy and Separate in Imgages in Test and Train Folder
def copy_images_to_train_test(df_train, df_test, subfolder, class_id=2):
    path = os.getcwd()
    #print(path)
    path_suffix = 'c' + str(class_id) + '/'
    
    create_train_test_folders(subfolder, class_id)
    df_train = df_train.query('ClassId == @class_id')
    df_test = df_test.query('ClassId == @class_id')
    
    print('copying images to folders...')

    for i in range(len(df_train)):
        # for training data
        origin_train_path = path + '/data/train_images/'
        source_file_train = df_train.iloc[i,1]
        #print(source_file_train)
        target_directory_train = path + '/data/' + subfolder + '/train/' + path_suffix
        #print(origin_train_path)
        #print(target_directory_train)
            
        # Copy The Files
        shutil.copy2(origin_train_path + source_file_train, 
                     target_directory_train + source_file_train)
        try:
            # for testing data
            origin_test_path = path + '/data/train_images/'
            source_file_test = df_test.iloc[i,1]
            target_directory_test = path + '/data/' + subfolder + '/test/' + path_suffix

            shutil.copy2(origin_test_path + source_file_test, 
                         target_directory_test + source_file_test)
        except:
            continue
    print(f'Images successfully copied to {subfolder}!')
    print()
    


def create_mask_image(image, image_id, image_dimension, encoded_pixels, inverse_masks):
    # path = os.getcwd()
    # target_directory = '/data/segmentation/train_mask/c1/'
    image_name = 'mask_' + image_id
    
    # os.chdir(path + target_directory)
    #print(target_directory.split('/')[0])
    #print(path.joinpath(target_directory.split('/')[0], image_name))
    if inverse_masks:
        mask = mask_conversion.create_mask_with_class_id_inverted(image_dimension,class_id=2,encoded_pixels=encoded_pixels)
    else:
        mask = mask_conversion.create_mask_with_class_id(image_dimension,class_id=2,encoded_pixels=encoded_pixels)
    mask *= 255
    
    written = cv2.imwrite(image_name, mask)
    #print(written)
    
    
    
def generate_mask_images(df, class_id, image_dimension, train=True, inverse_masks=False):
    """generates and saves mask images for `class_id`
    """
    print(f'generating mask images for ClassId {class_id} images...')
    image_ids = df.query('ClassId == @class_id').ImageId

    path = os.getcwd()
    path_suffix = 'c' + str(class_id) + '/'
    #print(path)
    if train:
        target_directory = '/data/segmentation/train_mask/' + path_suffix
    else:
        target_directory = '/data/segmentation/test_mask/' + path_suffix
    # switch to target directory for saving process
    os.chdir(path + target_directory)

    for image_id in image_ids:
        if train:
            image = cv2.imread('/data/segmentation/train/' + path_suffix + image_id)
        else:
            image = cv2.imread('/data/segmentation/test/' + path_suffix + image_id)
        encoded_pixels = df.query('ImageId == @image_id and ClassId == @class_id')[['EncodedPixels']]
        encoded_pixels = encoded_pixels.EncodedPixels.values[0]
        create_mask_image(image, image_id, image_dimension, encoded_pixels, inverse_masks)
        
    # switch back to home directory
    os.chdir(path)
    
    print('mask images successfully generated!')
    print()
  

import albumentations as A

augment = A.Compose([
    #A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.OneOf([
    #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    #     A.GridDistortion(p=0.5),
    #     A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
    #     ], p=0.8),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8)
])
    

def augement_images_and_masks(image_ids, num_augmentations, class_id):
    print(f'beginning augmentation for ClassId {class_id}...')
    start = time.time()
    
    path = os.getcwd()
    path_suffix = 'c' + str(class_id) + '/'
    
    target_directory_image = '/data/segmentation/train_aug/' + path_suffix
    target_directory_mask = '/data/segmentation/train_mask_aug/' + path_suffix
    
    i = 1
    
    while i <= num_augmentations:
        #print(i)
        number = random.randint(0, len(image_ids) -1)
        image_id = image_ids[number]
        mask_id = 'mask_' + image_id
        #print(image_id, mask_id)
        
        original_image = cv2.imread('/data/segmentation/train/' + path_suffix + image_id)
        #print(type(original_image))
        original_mask = cv2.imread('/data/segmentation/train_mask/' + path_suffix + mask_id)
        
        augmented = augment(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']
        
        os.chdir(path + target_directory_image)
        written = cv2.imwrite('aug_' + str(i) + '_' + image_id, transformed_image)
        #print(written)
        
        os.chdir(path + target_directory_mask)
        written = cv2.imwrite('aug_' + str(i) + '_' + mask_id, transformed_mask)
        #print(written)
        
        os.chdir(path)
        
        i += 1
    end = time.time()
    print(f'augmented {num_augmentations} images of ClassId {class_id}')
    print('time required for augmentation:', end - start)
    print()
    
    
    
def prepare_data_for_class_id(df, image_dimension, seed, class_id, inverse_masks):
    """combines all data preparations:
    - creating required folder structure
    - copying images to folders according to train-test-split using `seed`
    - generating masks for all images of `class_id`
    - augmenting images and corresponding masks

    Input parameters:
    df            - data frame that contains all defects and the `FilePaths` to all images
    seed          - seed for `train_test_split`
    class_id      - id of defect class
    inverse_masks - if `True`, defect pixels will be white, pixels without defect will be black
    """

    print('Starting data preparations')
    print('-----'*10)

    start = time.time()
    # split data into train and test
    df_train, df_test = create_train_test_dfs(df, seed)

    subfolder = 'segmentation'
    # sort images according to train-test-split
    copy_images_to_train_test(df_train, df_test, subfolder, class_id)

    # generate mask images for train and test data
    generate_mask_images(df, class_id, image_dimension, train=True, inverse_masks=inverse_masks)
    generate_mask_images(df, class_id, image_dimension, train=False, inverse_masks=inverse_masks)

    # augment images and masks where needed
    image_ids = df_train.query('ClassId == @class_id').ImageId.values
    num_augmentations = max(df_train.groupby('ClassId').count().ImageId)

    augement_images_and_masks(image_ids=image_ids, num_augmentations=num_augmentations, class_id=class_id)
    
    end = time.time()

    print('data successfully prepared for the model!')
    print('time elapsed:', end-start)