from scipy.io import wavfile
from transform import *
from sklearn.mixture import GaussianMixture
import util
import numpy as np
import argparse
from librosa import resample

def is_wav_file(filename):
    '''
    Check if the file is '.wav' file or not
    :param filename: file name to check out
    :return: True if the file is '.wav' file
    '''
    return any(filename.endswith(extension) for extension in [".wav"])

def load_wav_file(filename):
    '''
    Load wav file using filename
    :param filename: file name to open
    :return: sample rate and data of the wav file
    '''
    sample_rate, data = wavfile.read(filename)
    return sample_rate, data

def load_data(filepath):
    '''
    Load wav file and pre-process it to correct shape and type
    :param filepath: file path of wav file
    :return: mfcc data
    '''
    sample_rate, data = load_wav_file(filepath)
    if data.dtype != np.floating:
        data = data.astype(np.floating)
        
    if sample_rate != 16000:
        data = resample(data, sample_rate, 16000)
    
    # scaling needed -> not flatten yet
    data = mfcc(data, sample_rate, normalization=1, logscale=1, delta=0)
    return data

def paste_slices(tup):
    pos, w, max_w = tup
    dest_min = max(pos, 0)
    dest_max = min(pos+w, max_w)
    src_min = -min(pos,0)
    src_max = max_w - max(pos+w, max_w)
    src_max = src_max if src_max != 0 else None
    return slice(dest_min, dest_max), slice(src_min, src_max)

def paste(dest, src, loc):
    loc_zip = zip(loc, src.shape, dest.shape)
    dest_slices, src_slices = zip(*map(paste_slices,loc_zip))
    dest[dest_slices] += src[src_slices]

parser = argparse.ArgumentParser(description="Acoustic Scene Classification Using Gaussian Mixture Model")
parser.add_argument('--input_file', type=str, default="./example/example1.wav", help="Input file path")
parser.add_argument('--model', type=str, default="./model/gmm.json", help="Model file(json) to load")

opt = parser.parse_args()

# Labels of your Machine
labels = ['calm', 'traffic', 'noisy', 'tv']

filepath = opt.input_file
estimator = util.load_model(opt.model)

n_classes, input_shape = estimator.means_.shape

# 13 is the number of mfcc feature
input = np.zeros([input_shape//13, 13])
wav = load_data(filepath)

paste(input, wav, (0,0))
input = np.asarray([input.flatten()])

output = estimator.predict(input)

for item in enumerate(labels):
    if item[0] == output:
        print(item[1])
