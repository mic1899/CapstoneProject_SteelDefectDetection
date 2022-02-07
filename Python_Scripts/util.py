import pandas as pd

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