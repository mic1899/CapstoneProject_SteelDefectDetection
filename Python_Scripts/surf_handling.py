import numpy as np
import pandas as pd
import pathlib
from skimage import io
import time

def rank_keypoint(kp_dict, ranking_number):
    """Returns a keypoint at position `ranking_number` from a given dictionary of keypoints (`kp_dict`).
    
    Input variables:
    kp_dict        - dictionary with entries `size` (of keypoint) and `keypoint` (object)
    ranking_number - position of ranked keypoint (by size)
    """
    df = pd.DataFrame.from_dict(kp_dict)
    df = df.sort_values(by='size', ascending=False)

    # get number of kp in the dictionary to create ranking
    keypoints_in_dict = len(kp_dict['keypoint'])
    
    # create array with numbers from 1 to `keypoints_in_dict`
    ranking_numbers = np.linspace(1, keypoints_in_dict, keypoints_in_dict)
    # turn it into a data frame column
    ranking = pd.DataFrame(ranking_numbers, columns=['Ranking'])
    
    # merge with sorted df after index reset
    df.reset_index(drop=True, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    df_with_ranking = pd.concat([df, ranking], axis=1, ignore_index=False)

    # get entry at `ranking_number`
    df_ranked = df_with_ranking.query('Ranking == @ranking_number')

    return df_ranked.keypoint


def build_keypoint_dict(keypoints):
    """Turns a list of keypoint objects into a dictionary with entries `size` and `keypoint`.
    """
    kp_dict = {'size': [],
               'keypoint': []
              }

    for kp in keypoints:
        kp_dict['size'].append(kp.size)
        kp_dict['keypoint'].append(kp)
        
    return kp_dict


def get_ranked_keypoint(keypoints, ranking_number):
    """Returns a keypoint at position `ranking_number` and 0 if no keypoint exists for the ranking position.
    
    Input variables:
    keypoints      - list of keypoint objects from SURF algorithm
    ranking_number - position of ranked keypoint (by size)
    """
    # convert keypoints into dictionary with added size
    kp_dict = build_keypoint_dict(keypoints)
    
    # check, whether there is a keypoint for the required `ranking_number`
    if len(kp_dict['keypoint']) >= ranking_number:
        ranked_keypoint = rank_keypoint(kp_dict, ranking_number)
        
        # the keypoint needs to be extracted by its index
        return ranked_keypoint[ranking_number - 1]
    else:
        return 0
    
    
def get_keypoint_x(keypoints, ranking_number):
    """Function to calculate the x-coordinate of a keypoint at rank `ranking_number`.
    Returns 0 if no keypoint available for `ranking_number`.
    
    Input variables:
    keypoints      - list of keypoint objects from SURF algorithm
    ranking_number - position of ranked keypoint (by size)
    """
    if len(keypoints) != 0:
        ranked_keypoint = get_ranked_keypoint(keypoints, ranking_number)
        # validate that the keypoint is not empty
        if ranked_keypoint:
            return ranked_keypoint.pt[0]
        else:
            return 0
    else:
        return 0
    
    
def get_keypoint_y(keypoints, ranking_number):
    """Function to calculate the y-coordinate of a keypoint at rank `ranking_number`.
    Returns 0 if no keypoint available for `ranking_number`.
    
    Input variables:
    keypoints      - list of keypoint objects from SURF algorithm
    ranking_number - position of ranked keypoint (by size)
    """
    if len(keypoints) != 0:
        ranked_keypoint = get_ranked_keypoint(keypoints, ranking_number)
        # validate that the keypoint is not empty
        if ranked_keypoint:
            return ranked_keypoint.pt[1]
        else:
            return 0
    else:
        return 0
    
    
def get_keypoint_size(keypoints, ranking_number):
    """Function to calculate the size of a keypoint at rank `ranking_number`.
    Returns 0 if no keypoint available for `ranking_number`.
    
    Input variables:
    keypoints      - list of keypoint objects from SURF algorithm
    ranking_number - position of ranked keypoint (by size)
    """
    if len(keypoints) != 0:
        ranked_keypoint = get_ranked_keypoint(keypoints, ranking_number)
        # validate that the keypoint is not empty
        if ranked_keypoint:
            return ranked_keypoint.size
        else:
            return 0
    else:
        return 0
    
    
def add_keypoint_parameters(df, max_rank=50):
    """Adds columns to df for keypoint parameters `x`, `y`, and `size` up to `max_rank`.
    Run time for 50 keypoints: ~ 35 minutes
    """
    total_time = 0

    for i in range(1,max_rank + 1):
        # initialize temporary parameters
        name_x = 'kp_x_' + str(i)
        name_y = 'kp_y_' + str(i)
        name_size = 'kp_size_' + str(i)
        x = pd.DataFrame(columns = [name_x])
        y = pd.DataFrame(columns = [name_y])
        s = pd.DataFrame(columns = [name_size])
        
        print('processing step ...', i)
        start = time.time()

        x[name_x] = df.keypoints.apply(lambda x: get_keypoint_x(x, i))
        y[name_y] = df.keypoints.apply(lambda x: get_keypoint_y(x, i))
        s[name_size] = df.keypoints.apply(lambda x: get_keypoint_size(x, i))
        
        # piece everything together before the next run
        df = pd.concat([df, x, y, s], axis=1, ignore_index=False)

        end = time.time()
        total_time += (end - start)

        print('processing time:', end-start)
    print('total processing time was:', total_time)
    print('average processing time per rank:', total_time / max_rank)
  
    return df


def add_keypoints_to_frame(df, surf_object):
    # prepare dictionary to gather data
    surf_images = {'keypoints': [],
                   'ImageId': [],
                   'NumberKP': []
                  }

    print('processing images...')
    start = time.time()

    for idx, image_id in enumerate(df.ImageId):
        surf_images['ImageId'].append(image_id)

        # `image` so far holds just the path to the image. Convert to image file
        image = io.imread("data/train_images/" + image_id)
        # Find keypoints and descriptors directly
        kp, des = surf_object.detectAndCompute(image, None)

        surf_images['keypoints'].append(kp)
        surf_images['NumberKP'].append(len(kp))
        if idx % 500 == 0 and idx != 0:
            print(f'image number {idx} processed...')

    end = time.time()
    print('processing done.')
    print('required time:', end - start)
    
    temp = pd.DataFrame.from_dict(surf_images)
    
    return df.merge(temp, on='ImageId')


def build_keypoints_from_list(train_images_list, surf_object):
    # prepare dictionary to gather data
    surf_images = {'keypoints': [],
                   'ImageId': [],
                   'NumberKP': []
                  }

    print('processing images...')
    start = time.time()

    for idx, image in enumerate(train_images_list):
        surf_images['ImageId'].append(image.name)
    
        # `image` so far holds just the path to the image. Convert to image file
        image = io.imread("data/train_images/" + image.name)
        # Find keypoints and descriptors directly
        kp, des = surf.detectAndCompute(image, None)

        surf_images['keypoints'].append(kp)
        surf_images['NumberKP'].append(len(kp))
        if idx % 500 == 0 and idx != 0:
            print(f'image number {idx} processed...')

    end = time.time()
    print('processing done.')
    print('required time:', end - start)
    
    temp = pd.DataFrame.from_dict(surf_images)
    
    return temp