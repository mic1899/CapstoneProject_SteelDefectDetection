import pandas as pd
import numpy as np


""" DECODING """

def get_column(pixel, rows=256): 
    """
    Input variables:
    pixel - String
    """
    return (int(pixel) - 1) // rows

def get_row(pixel, rows=256):
    """
    Input variables:
    pixel - String
    """
    return (int(pixel) - 1) % rows

def get_column_row(pixel, rows=256):
    """
    Input variables:
    pixel - String
    """
    column = get_column(pixel, rows)
    row = get_row(pixel, rows)

    return column, row


# split up encoded pixels in pairs of [start_pixel, pixel_length]
def get_pixel_pairs(encoded_pixels):
    """Returns a list of pixel pairs from the `encoded_pixels`.
    
    Input variables:
    encoded_pixels - String with encoded pixel pairs of shape (start_pixel, pixel_length)
    """
    i = 0 # running variable
    temp = [0,0]
    pairs = []
    # iterate over all `encoded_pixels` items to build pairs
    for pixel in encoded_pixels.split():
        temp[i] = pixel
        if i%2 != 0: # re-initialise after a pair is complete
            pairs.append(temp)
            i = 0
            temp = [0,0]
            continue # skip increment of i
        i += 1
    return pairs


def create_mask_with_class_id_inverted(image_dimension, class_id, encoded_pixels, image=True):
    """set specific values in a null-matrix to `class_id` and returns the filled mask.
    
    Input variables:
    image_dimension - tupel of the dimension of the image or matrix
    class_id        - the class that will be set at `encoded_pixels`
    encoded_pixels  - String with encoded pixel pairs of shape (start_pixel, pixel_length)
    image           - for testing purposes set to false
    rows            - for testing purposes set to shape[0] of testing matrix
    """
    length_sum = 0
    # initialise mask
    mask = np.ones(image_dimension)
    rows = image_dimension[0]
    
    for pair in get_pixel_pairs(encoded_pixels):
        # fetch column and row indices
        if image:
            column, row = get_column_row(pair[0])
        else: # for testing purposes
            column, row = get_column_row(pair[0],rows)
        # testing
        length_sum += int(pair[1])
        
        # set all pixels of the respective pair to `class_id`
        for pixel in range(int(pair[1])):
            # write into same column
            if (row + pixel) < rows:
                mask[row + pixel][column] = 0  
                
            else: # for column changes
                # compute required column and row shifts for `class_id` placement
                col_shift = (row + pixel) // rows 
                row_shift = pixel - col_shift * rows

                # write `class_id` to shifted position
                mask[row + row_shift][column + col_shift] = 0

    return mask


def create_mask_with_class_id(image_dimension, class_id, encoded_pixels, image=True):
    """set specific values in a null-matrix to `class_id` and returns the filled mask.
    
    Input variables:
    image_dimension - tupel of the dimension of the image or matrix
    class_id        - the class that will be set at `encoded_pixels`
    encoded_pixels  - String with encoded pixel pairs of shape (start_pixel, pixel_length)
    image           - for testing purposes set to false
    rows            - for testing purposes set to shape[0] of testing matrix
    """
    length_sum = 0
    # initialise mask
    mask = np.zeros(image_dimension)
    rows = image_dimension[0]
    
    for pair in get_pixel_pairs(encoded_pixels):
        # fetch column and row indices
        if image:
            column, row = get_column_row(pair[0])
        else: # for testing purposes
            column, row = get_column_row(pair[0],rows)
        # testing
        length_sum += int(pair[1])
        
        # set all pixels of the respective pair to `class_id`
        for pixel in range(int(pair[1])):
            # write into same column
            if (row + pixel) < rows:
                mask[row + pixel][column] = int(class_id)  
                
            else: # for column changes
                # compute required column and row shifts for `class_id` placement
                col_shift = (row + pixel) // rows 
                row_shift = pixel - col_shift * rows

                # write `class_id` to shifted position
                mask[row + row_shift][column + col_shift] = int(class_id)

    return mask


def decode_pixel(image_dimension, class_id, encoded_pixels, image=True):
    mask = create_mask_with_class_id(image_dimension, class_id, encoded_pixels, image=True)
    return mask


def decode_pixel_inverted(image_dimension, class_id, encoded_pixels, image=True):
    mask = create_mask_with_class_id_inverted(image_dimension, class_id, encoded_pixels, image=True)
    return mask


###################################################################################################

""" ENCODING """

def get_encoding_length(column, start, class_id):
    """returns the length of a `class_id` encoding from `start`
    
    Input variables:
    column   - column where an encoding was found
    start    - start index of the encoding
    class_id - `class_id` of the encoding that should be marked
    """
    same_encoding = True
    length = 0
    i = start
    
    while same_encoding and i < column.shape[0]:
        pixel = column[i]
        if pixel == class_id:
            length += 1
            i += 1
        else:
            # as soon as a pixel doesn't match `class_id`, the encoding ends
            same_encoding = False
            
    return length


def find_start_pixels(column, class_id=1):
    """returns a list of `start_pixels` of `class_id` encodings in `column`
    """
    start_pixels = []
    idx = 0
    
    while idx < column.shape[0]:
        pixel = column[idx]
        if pixel == class_id:
            # as soon as we find a pixel from `class_id`, we add it to the list
            start_pixels.append(idx)
            # skip all pixels of the same encoding
            shift = get_encoding_length(column, idx, class_id)
            idx += shift
        else:
            idx += 1
            
    return start_pixels


def get_encoding_list(mask, class_id=1):
    """returns a list of pairs of (`starting_pixel`, `pixel_length`) for a `class_id` encoding in `mask`
    """
    list_of_encodings = []
    
    for col_idx in range(mask.shape[1]):
        column = mask[:, col_idx]
        # get all starting points for an encoding in `column`
        start_pixels = find_start_pixels(column, class_id)
        
        for start_pixel in start_pixels:
            length_encoding = get_encoding_length(column, start_pixel, class_id)
            # adjust start pixels for their respective row
            start = start_pixel + col_idx * column.shape[0]
            
            list_of_encodings.append(str(start + 1))
            list_of_encodings.append(str(length_encoding))
            
    return list_of_encodings


def encode_pixel(mask, class_id=1):
    """returns a string with pairs of (`starting_pixel`, `pixel_length`) for a `class_id` encoding in `mask`
    """
    encoding = get_encoding_list(mask, class_id)
    return ' '.join(encoding)