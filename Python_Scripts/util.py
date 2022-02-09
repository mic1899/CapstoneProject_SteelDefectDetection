import pandas as pd
from skimage import io


def isolate_single_defects(temp):
    """Isolates all `ImageIds` from `train_complete.csv` that have more than 1 defect and drops them from `temp`.
    """
    df_complete = pd.read_csv('data/train_complete.csv')
    # Count occurrences of `ImageId` in df
    df_complete['count'] = df_complete.ImageId.apply(lambda x: df_complete['ImageId'].value_counts()[x])

    # isolate `ImageIds` for images with defect
    single_defects = df_complete.query('count > 1').ImageId.to_numpy()

    # get indices of `df_raw` for row dropping
    indices = []
    for idx, row in temp.iterrows():
        if row.ImageId in single_defects:
            indices.append(idx)

    temp.drop(indices, inplace=True)
    
    
def get_indices_for_class_id(y, class_id):
    """returns an array of indices where vector y (e.g. `y_train`, `y_test`) matches `class_id`.
    
    Input parameters:
    y        - vector that contains class ids (y_train or y_test)
    class_id - of defect that should be isolated
    """
    pos_of_class_id = (y == class_id)
    indices = pos_of_class_id[pos_of_class_id].index.values
    return indices


def get_black_columns(image, threshold=5):
    """returns the number of columns that are black, i.e. the sum of the 3 color planes does not exceed the (number of rows) * 3 * `threshold`.
    """
    num_columns = 0
    
    for column in range(image.shape[1]):
        color_sum = image[:, column].sum()
        
        if color_sum <= image.shape[0] * 3 * threshold:
            num_columns += 1
            
    return num_columns


def add_blackness_attributes_for_single_class(image_df, y_train, folder_extension, class_id):
    """returns the `image_df` extended by columns `BlackColumns` and `PercentageBlack` for a single class.
    
    Input parameters:
    image_df         - data frame that includes `ImageIds`
    y_train          - vector with `ClassIds`
    folder_extension - the subfolder in 'data/' where pictures are located
    class_id         - the `ClassId` for which attributes shall be added
    """
    black_columns = []
    black_columns_percentage = []

    for image_id in image_df.ImageId:
        image = io.imread('data/' + folder_extension + '/' + image_id)
        black_columns.append(get_black_columns(image))
        black_columns_percentage.append(get_black_columns(image) / image.shape[1])

    temp = pd.DataFrame(list(zip(black_columns, black_columns_percentage)), 
                        index=get_indices_for_class_id(y_train, class_id), 
                        columns = ['BlackColumns', 'PercentageBlack'])

        
    image_df = pd.merge(image_df, temp, left_index=True, right_index=True)

    return image_df


def add_blackness_attributes(image_df, folder_extension):
    """returns the `image_df` extended by columns `BlackColumns` and `PercentageBlack`.
    
    Input parameters:
    image_df         - data frame that includes `ImageIds`
    folder_extension - the subfolder in 'data/' where pictures are located
    """
    black_columns = []
    black_columns_percentage = []

    for image_id in image_df.ImageId:
        image = io.imread('data/' + folder_extension + '/' + image_id)
        black_columns.append(get_black_columns(image))
        black_columns_percentage.append(get_black_columns(image) / image.shape[1])
    
    temp = pd.DataFrame(list(zip(black_columns, black_columns_percentage)), 
                        columns = ['BlackColumns', 'PercentageBlack'])
        
    image_df = pd.merge(image_df, temp, left_index=True, right_index=True)

    return image_df