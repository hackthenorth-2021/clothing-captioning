import argparse
import os
from model import get_model
from generator import get_generator
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import ModelCheckpoint

import subprocess
subprocess.call(['pip', 'install', 'scikit-learn'])
subprocess.call(['pip', 'install', 'matplotlib'])
subprocess.call(['pip', 'install', 'h5py<3.0.0'])

from azureml.core import Run
run = Run.get_context()

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_base_path",
        required=True
    )
    parser.add_argument(
        "--hdc_img_size",
        required=True
    )
    args = parser.parse_args()
    return args

def train(args):
    clear_session()

    train_path = os.path.join(args.dataset_base_path, 'train')
    train_label = "train.json"

    val_path = os.path.join(args.dataset_base_path, 'val')
    val_label = "val.json"

    model_saving_callback = ModelCheckpoint("output", monitor='val_loss', save_best_only=True, mode='auto')

    callbacks = [model_saving_callback]

    generator = get_generator(train_path, train_label)
    model.fit(x=generator,
              epochs=10,
              callbacks=callbacks)

if __name__ == '__main__':
    args = read_args()
    model = get_model(args)

