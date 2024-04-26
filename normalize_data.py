import os
import numpy as np
import torch
from helpers import normalize

TRAIN_SAMPLES_PATH = 'MagnaTagATune/samples/train'
VAL_SAMPLES_PATH = 'MagnaTagATune/samples/val'

os.mkdir('MagnaTagATune/samples_norm')
os.mkdir('MagnaTagATune/samples_norm/train')

for folder_name in os.listdir(TRAIN_SAMPLES_PATH):
    os.mkdir(f'MagnaTagATune/samples_norm/train/{folder_name}')

    for file_name in os.listdir(f'{TRAIN_SAMPLES_PATH}/{folder_name}'):
        sample = torch.from_numpy(np.load(f"{TRAIN_SAMPLES_PATH}/{folder_name}/{file_name}"))
        normalized_sample = normalize(sample)
        np.save(f'MagnaTagATune/samples_norm/train/{folder_name}/{file_name}', normalized_sample)

os.mkdir('MagnaTagATune/samples_norm/val')

for folder_name in os.listdir(f'{VAL_SAMPLES_PATH}'):
    os.mkdir(f'MagnaTagATune/samples_norm/val/{folder_name}')

    for file_name in os.listdir(f'{VAL_SAMPLES_PATH}/{folder_name}'):
        sample = torch.from_numpy(np.load(f"{VAL_SAMPLES_PATH}/{folder_name}/{file_name}"))
        normalized_sample = normalize(sample)
        np.save(f'MagnaTagATune/samples_norm/val/{folder_name}/{file_name}', normalized_sample)
