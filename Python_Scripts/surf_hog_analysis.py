import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

"""For model performance"""

def get_false_prediction_indices(y_test, y_pred):
    # extract indices where predictions were incorrect
    false_predictions = (y_pred!=y_test)
    false_prediction_indices = false_predictions[false_predictions].index.values
    
    return false_prediction_indices


def get_true_labels_of_false_predictions(y_test, y_pred, false_prediction_indices):
    # keep labels of false predictions
    y_pred_false =y_pred[(y_pred!=y_test)]
    true_labels = pd.DataFrame(y_pred_false, index=false_prediction_indices, columns=['ClassId_predicted'])
    return true_labels


def get_false_predicted_images(df_complete, df_model, false_prediction_indices, true_labels):
    # extract all rows from `df_model` where the predcitions were incorrect
    false_predicted_images = df_model.join(true_labels) \
                                .loc[false_prediction_indices][['ImageId','ClassId', 'ClassId_predicted']]
    # add additional information needed to find the correponding pictures
    false_predicted_images = false_predicted_images.merge(df_complete[['FilePath','ImageId']], on = 'ImageId')
    
    return false_predicted_images


def print_misclassifications(false_predicted_images, number_images):
    # create random index for `number_images`
    random_index = np.round(np.random.rand(number_images) * len(false_predicted_images.ImageId)) + 1

    for i in range(number_images):
        # gather required info to retrieve image and label the plots
        file_path_to_image = false_predicted_images['FilePath'][random_index[i]]
        class_id = false_predicted_images['ClassId'][random_index[i]]
        image_id = false_predicted_images['ImageId'][random_index[i]]
        class_id_pred = int(false_predicted_images['ClassId_predicted'][random_index[i]])
        
        # read-in the image
        img = io.imread(file_path_to_image)
        plt.figure(figsize=(18, 10))
        
        ax = plt.subplot(number_images, 1, i + 1)
        plt.imshow(img)
        
        plt.title(f'Image ID: {image_id} | True ClassId: {class_id} | Predicted ClassId: {class_id_pred}', fontsize=16);
        plt.axis("off")
        
        
def print_false_classifications(df_complete, df_model, y_test, y_pred, number_images=5):
    """prints out `number_images` different, randomly selected images that were misclassified in the model.
    
    Input parameters:
    df_complete   - data frame that contains the `FilePath` of all images
    df_model      - data frame that was used in the model
    y_test        - from `train_test_split`
    y_pred        - predicted labels by the model
    number_images - number of different, randomly selected images from the model
    """
    
    # get indices of misclassified images
    false_prediction_indices = get_false_prediction_indices(y_test, y_pred)
    # get `true_labels` of the misclassified images
    true_labels = get_true_labels_of_false_predictions(y_test, y_pred, false_prediction_indices)
    
    # build a data frame with all necessary information
    false_predicted_images = get_false_predicted_images(df_complete, df_model, false_prediction_indices, true_labels)
    # print randomly selected images from misclassified images
    print_misclassifications(false_predicted_images, number_images)
    

"""For general purpose"""


def print_batch(df_with_filepath, class_ids, blackness=False, show_keypoints=False, number_images=5):
    # create random index for `number_images`
    random_index = np.array(np.random.rand(number_images) * len(df_with_filepath.ImageId) + 1, dtype='int')

    for i in range(number_images):
        # gather required info to retrieve image and label the plots
        file_path_to_image = df_with_filepath['FilePath'].iloc[random_index[i]]
        class_id = class_ids.iloc[random_index[i]]
        image_id = df_with_filepath['ImageId'].iloc[random_index[i]]
        if blackness:
            blackness = df_with_filepath['PercentageBlack'].iloc[random_index[i]]
        if show_keypoints:
            keypoints = df_with_filepath['NumberKP'].iloc[random_index[i]]


        
        # read-in the image
        img = io.imread(file_path_to_image)
        plt.figure(figsize=(25, 14))
        
        ax = plt.subplot(number_images, 1, i + 1)
        plt.imshow(img)
        title = f'Image ID: {image_id} | ClassId: {class_id}'
        if blackness:
            title += f' | Percentage Black: {blackness}'
        if show_keypoints:
            title += f' | Number Keypoints: {keypoints}'
        plt.title(title, fontsize=16);
        plt.axis("off")
        
