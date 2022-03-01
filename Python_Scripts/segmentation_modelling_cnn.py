import segmentation_models as sm
import tensorflow as tf
from keras.losses import binary_crossentropy

import numpy as np
import pandas as pd
import glob
import os
import cv2

import matplotlib.pyplot as plt

# self-written scripts
import sys
sys.path.insert(0, 'Python_Scripts')

import data_preparation_cnn



def get_images(class_id, size_x, size_y):
    images = []
    path_suffix = 'c' + str(class_id) + '/'

    for directory_path in glob.glob('data/segmentation/test/' + path_suffix):
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.jpg"))):

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (size_y, size_x))

            images.append(img)
            
    #Convert list to array for machine learning processing        
    images = np.array(images)
    
    return images



def get_masks(class_id, size_x, size_y):
    images = []
    path_suffix = 'c' + str(class_id) + '/'

    for directory_path in glob.glob('data/segmentation/test_mask/' + path_suffix):
        for img_path in sorted(glob.glob(os.path.join(directory_path, "*.png"))):

            img = cv2.imread(img_path, 0)       
            img = cv2.resize(img, (size_y, size_x))

            images.append(img)

    #Convert list to array for machine learning processing        
    images = np.array(images)
    
    return images



def prepare_input_variables(class_id, train_images, train_masks, size_x, size_y):
    # get preprocessing for `EfficientNetB5`
    preprocess_input = sm.get_preprocessing('efficientnetb5')
    
    # prepare variables
    x_train = preprocess_input(train_images)
    y_train = np.expand_dims(train_masks, axis=3)

    x_val = get_images(class_id=class_id, size_x=size_x, size_y=size_y)
    x_val = preprocess_input(x_val)
    y_val = get_masks(class_id=class_id, size_x=size_x, size_y=size_y)
    y_val = np.expand_dims(y_val, axis=3) #May not be necessary.. leftover from previous code 

    return x_train, y_train, x_val, y_val



"""METRIC AND LOSS FUNCTION FOR DICE-COEFFICIENT"""

def dice_coef(y_true,y_pred):
    y_true_f=tf.reshape(tf.dtypes.cast(y_true, tf.float32),[-1])
    y_pred_f=tf.reshape(tf.dtypes.cast(y_pred, tf.float32),[-1])
    intersection=tf.reduce_sum(y_true_f*y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return (1-dice_coef(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return binary_crossentropy(y_true, y_pred) + (1-dice_coef(y_true, y_pred))



def build_compiled_model(size_x, size_y, metric_for_model, optimizer):
    # set the correct framework for the model to work
    sm.set_framework('tf.keras')
    sm.framework()
    
    # define model
    model = sm.Unet('efficientnetb5',
                    input_shape=(size_x, size_y, 3),
                    classes=1,
                    activation='sigmoid',
                    encoder_weights='imagenet',
                    encoder_freeze=True
                   )
    if metric_for_model == 'dice':
        model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coef])
    else:
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[metric_for_model])
    
    return model


def get_history_from_mask_model(class_id, size_x, size_y, metric_for_model, epochs, optimizer):
    tf.keras.backend.clear_session()
    # load images and masks as input for model
    train_images, train_masks = data_preparation_cnn.get_resized_image_and_mask_lists(class_id=class_id, 
                                                                              size_x=size_x, 
                                                                              size_y=size_y)

    # build input variables for model
    x_train, y_train, x_val, y_val = prepare_input_variables(class_id, 
                                                             train_images, 
                                                             train_masks, 
                                                             size_x, 
                                                             size_y)
        
    # create a compiled model
    model = build_compiled_model(size_x, size_y, metric_for_model, optimizer)
    
    print(f'beginning training with masks')
    # fit the model
    history = model.fit(x_train, 
                        y_train,
                        batch_size=32, 
                        epochs=epochs,
                        verbose=True,
                        validation_data=(x_val, y_val)
                       )
        
    # save model
    model_name = 'models/class' + str(class_id) + '_mask_generator.h5'
    model.save_weights(model_name)
    
    return history, model



def build_mask_generating_models(size_x, size_y, metric_for_model, epochs, optimizer):
    
    losses = []
    losses_val = []
    
    metric_results = []
    metric_results_val = []
    
    models = {}
    
    for class_id in [1,2,3,4]:
        print(f'building model for defect class {class_id}...')
        
        
        # extract the history for each model
        history, model = get_history_from_mask_model(class_id, 
                                                     size_x, size_y, 
                                                     metric_for_model, 
                                                     epochs, 
                                                     optimizer)
        
        losses.append(history.history['loss'])
        losses_val.append(history.history['val_loss'])
        if metric_for_model == 'dice':
            pass
        else:
            metric_results = history.history[metric_for_model]
            metric_results_val = history.history['val_' + metric_for_model]
        
        models[str(class_id)] = model
        
        print('model successfully generated and saved to file!')
        print('-----'*10)
        print()
    
    return models



def load_and_compile_cnn_models(size_x, size_y, metric):
    """returns a list of compiled models from our `build_mask_generating_models` function with their respective weights.
    """
    models = []
    
    for i in [1,2,3,4]:
        model_name = 'models/class' + str(i) + '_mask_generator.h5'
        
        model = build_compiled_model(size_x, size_y, metric)
        
        model.load_weights(model_name)
        
        models.append(model)
        
    return models



def load_and_compile_dice_models_from_capstone():
    """returns a list of compiled models from the capstone project using the dice coefficient
    """
    models = []
    size_x = 128
    size_y = 512
    learning_rate = 0.005
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    
    for i in [1,2,3,4]:
        print(f'Loading model {i}...')
        model_name = 'models/class_' + str(i) + '_20e_LR005.h5'
        
        model = build_compiled_model(size_x, size_y, 'dice', optimizer)
        
        model.load_weights(model_name)
        
        models.append(model)
        
    return models
        
    

"""FOR ACCURACY CALCULATIONS"""

def get_pred_dict(models, images):
    """returns a dictionary with predictions on `images` using `models`. Dictionary contains keys '1' - '4' which each holds 
    the predictions to all `images` for the respective `model`.
    
    input parameters:
    models - list of 4 models
    images - images that shall be predicted on
    """
    
    print('making predictions...')
    tf.keras.backend.clear_session()

    pred_model1 = models[0].predict(images)
    pred_model2 = models[1].predict(images)
    pred_model3 = models[2].predict(images)
    pred_model4 = models[3].predict(images)

    pred_dict = {
        '1': pred_model1,
        '2': pred_model2,
        '3': pred_model3,
        '4': pred_model4
    }
    
    return pred_dict



def calculate_sums(pred_dict, class_id):
    
    sums = {}
    temp = []
    for key in pred_dict.keys():
        # print(type(key))
        for prediction in range(len(pred_dict[key])): 
            temp.append(np.round(pred_dict[key][prediction]).sum())
                            
        sums[key] = temp
        temp = []
    
    return sums



def build_df(sums, class_id):
    df = pd.DataFrame()
    df = df.assign(a=sums['1']).assign(b=sums['2']).assign(c=sums['3']).assign(d=sums['4'])
    if class_id == 1:
        df['class2'] = (df.a < df.b).astype(int)
        df['class3'] = (df.a < df.c).astype(int)
        df['class4'] = (df.a < df.d).astype(int)
    if class_id == 2:
        df['class2'] = (df.b < df.a).astype(int)
        df['class3'] = (df.b < df.c).astype(int)
        df['class4'] = (df.b < df.d).astype(int)
    if class_id == 3:
        df['class2'] = (df.c < df.b).astype(int)
        df['class3'] = (df.c < df.a).astype(int)
        df['class4'] = (df.c < df.d).astype(int)
    if class_id == 4:
        df['class2'] = (df.d < df.b).astype(int)
        df['class3'] = (df.d < df.c).astype(int)
        df['class4'] = (df.d < df.a).astype(int)
    
    return df



def calculate_accuracy(df):
    print('calculating accuracy...')
    return 1 - ((df.class2.sum() + df.class3.sum() + df.class4.sum()) / len(df))



def get_images_for_prediction(class_id, size_x, size_y):
    preprocess_input = sm.get_preprocessing('efficientnetb5')
    
    images = get_images(class_id, size_x, size_y)
    images = preprocess_input(images)
    
    return images



def get_accuracy_for_class(models, class_id, size_x, size_y):
    print(f'Calculating accuracy for defect class {class_id}:')
    images = get_images_for_prediction(class_id, size_x, size_y)
    
    pred_dict = get_pred_dict(models, images)
    
    sums = calculate_sums(pred_dict, class_id)
    
    df = build_df(sums, class_id)
    
    accuracy = calculate_accuracy(df)
    
    return accuracy



def calculate_accuracies_for_masks(models, size_x, size_y):
    accuracies = []
    num_images = []
    weighted_acc = []


    for class_id in [1,2,3,4]:
        accuracies.append(get_accuracy_for_class(models, class_id, size_x, size_y))

        num_images.append(len(get_images_for_prediction(class_id, size_x, size_y)))


    for i in range(4):
        weighted_acc.append(num_images[i] * accuracies[i])


    print()
    print('Accuracies:', accuracies)
    print()
    print('Number of images per class:', num_images) 
    # print()
    # print('Weigthed accuracies:', weighted_acc)
    print()
    print('-----'*12)
    print('Weighted accuracy:', np.sum(weighted_acc) / np.sum(num_images))
    
    return accuracies, weighted_acc



"""FOR VISUALIZATION"""

def get_images_and_masks_for_display(size_x, size_y):
    
    images = []
    masks = []
    
    for class_id in [1,2,3,4]:
        images.append(get_images(class_id, size_x, size_y))
        masks.append(get_masks(class_id, size_x, size_y))
        
    return images, masks



def get_predictions_for_class(models, class_id, size_x, size_y):
    """returns a list of predicted mask images for `class_id`, one prediction for each model
    """
    print(f'Making predictions for defect class {class_id}:')
    
    images = get_images_for_prediction(class_id, size_x, size_y)
    
    pred_dict = get_pred_dict(models, images)
    
    predictions = []
    
    for key in pred_dict.keys():
        predictions.append(pred_dict[key])
        
    return predictions



def get_predictions_all_classes(models, size_x, size_y):
    """returns a dictionary with all predicted mask images for all `class_ids`.
    """
    predictions = {}
    
    for class_id in [1,2,3,4]:
        
        predictions[str(class_id)] = get_predictions_for_class(models, class_id, size_x, size_y)
    
    return predictions



def visualize_predictions(predictions, class_id, size_x, size_y):
    
    images, masks = get_images_and_masks_for_display(size_x, size_y)
    
    images = images[class_id - 1]
    masks  = masks[class_id - 1]
    
    # predictions = get_predictions_for_class(models, class_id, size_x, size_y)
    
    preds = predictions[str(class_id)]
    
    nrows = 5
    ncols = 6
    rows = list(range(nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*3,2*nrows))


    fig.suptitle(f'Performance of Mask Model from saved Model', fontsize=16)

    for row, a, b, c, d, e, f in zip(rows,
                                     images,
                                     masks, 
                                     preds[0], 
                                     preds[1], 
                                     preds[2], 
                                     preds[3]
                                    ):
        axs[row, 0].imshow(a)
        if row == 0:
            axs[row, 0].set_title(f'true image', fontsize=14)
        axs[row, 0].set_ylabel(f'Defect Id = {class_id}', fontsize=14)
        axs[row, 0].set_xticks([])
        axs[row, 0].set_yticks([])

        axs[row, 1].imshow(b)
        if row == 0:
            axs[row, 1].set_title('true mask', fontsize=14)
        axs[row, 1].set_axis_off()

        axs[row, 2].imshow(c)
        if row == 0:
            axs[row, 2].set_title('mask model 1', fontsize=14)
        axs[row, 2].set_axis_off()

        axs[row, 3].imshow(d)
        if row == 0:
            axs[row, 3].set_title('mask model 2', fontsize=14)
        axs[row, 3].set_axis_off()

        axs[row, 4].imshow(e)
        if row == 0:
            axs[row, 4].set_title('mask model 3', fontsize=14)
        axs[row, 4].set_axis_off()

        axs[row, 5].imshow(f)
        if row == 0:
            axs[row, 5].set_title('mask model 4', fontsize=14)
        axs[row, 5].set_axis_off()

        if row == nrows:
            break
            


def visualize_masks_across_defect_classes(predictions, size_x, size_y): 
    
    images, masks = get_images_and_masks_for_display(size_x, size_y)

    # generate random index numbers for different defect classes
    r1 = np.round(np.random.rand() * len(images[0]),0).astype(int)
    r2 = np.round(np.random.rand() * len(images[1]),0).astype(int)
    r3 = np.round(np.random.rand() * len(images[2]),0).astype(int)
    r4 = np.round(np.random.rand() * len(images[3]),0).astype(int)

    r_idxs = [r1, r2, r3, r4]
    print('image indices:', r_idxs)


    nrows = 4
    ncols = 5
    rows = list(range(nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5,2*nrows))

    imgs  = [images[0][r1], images[1][r2], images[2][r3], images[2][r4]]
    masks = [masks[0][r1],  masks[1][r2],  masks[2][r3],  masks[3][r4]]
    
    p1 = [predictions['1'][0][r1], predictions['1'][1][r1], predictions['1'][2][r1], predictions['1'][3][r1]]
    p2 = [predictions['2'][0][r2], predictions['2'][1][r2], predictions['2'][2][r2], predictions['2'][3][r2]]
    p3 = [predictions['3'][0][r3], predictions['3'][1][r3], predictions['3'][2][r3], predictions['3'][3][r3]]
    p4 = [predictions['4'][0][r4], predictions['4'][1][r4], predictions['4'][2][r4], predictions['4'][3][r4]]


    fig.suptitle(f'Performance of Mask Model', fontsize=24)
    

    for row, b, c, d, e, f in zip(rows, #imgs, 
                                     masks, p1, p2, p3, p4):

        # axs[row, 0].imshow(a)
        # if row == 0:
        #     axs[row, 0].set_title(f'true image', fontsize=14)
        # axs[row, 0].set_ylabel(f'DefectId = {row + 1}', fontsize=16)
        # axs[row, 0].set_xticks([])
        # axs[row, 0].set_yticks([])

        axs[row, 0].imshow(b)
        if row == 0:
            axs[row, 0].set_title('true mask', fontsize=16)
        axs[row, 0].set_axis_off()

        axs[row, 1].imshow(c)
        if row == 0:
            axs[row, 1].set_title('mask model 1', fontsize=16)
        axs[row, 1].set_axis_off()

        axs[row, 2].imshow(d)
        if row == 0:
            axs[row, 2].set_title('mask model 2', fontsize=16)
        axs[row, 2].set_axis_off()

        axs[row, 3].imshow(e)
        if row == 0:
            axs[row, 3].set_title('mask model 3', fontsize=16)
        axs[row, 3].set_axis_off()

        axs[row, 4].imshow(f)
        if row == 0:
            axs[row, 4].set_title('mask model 4', fontsize=16)
        axs[row, 4].set_axis_off()

        if row == nrows:
            break