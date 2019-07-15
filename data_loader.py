from __future__ import absolute_import

import numpy as np
from scipy.io import wavfile
from os import listdir
from os.path import join
from sklearn.model_selection import StratifiedKFold
import random
from transform import *
from tqdm import tqdm

def is_wav_file(filename):
    return any(filename.endswith(extension) for extension in [".wav"])

def load_wav_file(filename):
    """Loads an audio file and returns a float PCN-encoded array of samples."""
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data

def load_data(wave_dir):
    labels = listdir(wave_dir)
     
    ulabel = -1
    print(labels)
    dataset = {'data':[], 'target':[]}
    duration = []

    count = 0
    #DUE TO MEMORY ISSUE, RESTRICT THE NUMBER OF SAMPLES

    for label in labels:
        data_dir = join(wave_dir, label)

        if label == 'carm':
            ulabel = 0
        elif label == 'traffic':
            ulabel = 1
        elif label == 'noisy':
            ulabel = 2
        elif label == 'tv':
            ulabel = 3
        else:
            pass
        
        if ulabel == -1:
            #NO SUCH LABEL IN LABELS ERROR
            continue

        wav_filenames = listdir(data_dir)
        for filename in tqdm(wav_filenames):
            #RESTRICT MAX NUMBER
            '''
            if count == 6000:
                count = 0
                break
            '''

            filepath = join(data_dir, filename)
            sample_rate, data = load_wav_file(filepath)
            # SPECIFIED SAMPLING RATE (ALL IDENTICAL)
            if sample_rate == 16000:
                duration.append(data.shape[0])
                dataset['data'].append(data)
                dataset['target'].append(ulabel)
                count += 1
            
    #min_duration = min(duration)
    max_duration = max(duration)

    for i in tqdm(range (0, len(dataset['data']))):
        tmp = np.zeros(max_duration)
        tmp[:dataset['data'][i].shape[0]] = dataset['data'][i]
        dataset['data'][i] = tmp
        #dataset['data'][i] = dataset['data'][i][:min_duration]
        dataset['data'][i] = mfcc(dataset['data'][i], 16000, normalization=1, logscale=1, delta=0)
        dataset['data'][i] = dataset['data'][i].flatten()
    return dataset

def train_and_test(dataset):
    skf = StratifiedKFold(n_splits=5)
    train_index, test_index = next(iter(skf.split(dataset['data'], dataset['target'])))
    
    train_set = list(zip(np.asarray(dataset['data'])[train_index], np.asarray(dataset['target'])[list(train_index)]))
    random.shuffle(train_set)
    X_train = np.asarray([i for i, j in train_set])
    Y_train = np.asarray([j for i, j in train_set])
    X_test = np.asarray(dataset['data'])[test_index]
    Y_test = np.asarray(dataset['target'])[test_index]
    return X_train, Y_train, X_test, Y_test
