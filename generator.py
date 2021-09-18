import pandas as pd
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_generator(dir, label_file):
    '''
    Filters the list of data files and labels to only include successfully downloaded files

    :param dir: Directory to images and json label file
    :param label_file: Name of label file
    :return: DataFrameIterator to be used in model.fit as the data generator
    '''

    with open(join(dir,label_file), 'r') as f:
        label_dataframe = pd.DataFrame(json.load(f)['annotations'])
    label_dataframe['imageId'] = label_dataframe['imageId'].apply(lambda x: x+".jpg")
    img_files = [f for f in listdir(dir) if isfile(join(dir, f))]
    filtered_dataframe = label_dataframe[label_dataframe['imageId'].isin(img_files)]

    datagen = ImageDataGenerator(rescale=1./255.)
    generator = datagen.flow_from_dataframe(
        dataframe=filtered_dataframe,
        directory=dir,
        x_col="imageId",
        y_col="labelId",
        batch_size=4,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(512,512)
    )

    return generator