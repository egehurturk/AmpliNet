import pickle
import torch
from tqdm import tqdm
from scipy.io import wavfile
import argparse
import numpy as np
import math
from model import PedalNet
import librosa

def save(name, data):
    wavfile.write(name, 44100, data.flatten().astype(np.int16))

def greedy_split(arr, n, axis=0):
    """Greedily splits an array into n blocks.

    Splits array arr along axis into n blocks such that:
        - blocks 1 through n-1 are all the same size
        - the sum of all block sizes is equal to arr.shape[axis]
        - the last block is nonempty, and not bigger than the other blocks

    Intuitively, this "greedily" splits the array along the axis by making
    the first blocks as big as possible, then putting the leftovers in the
    last block.
    """
    length = arr.shape[axis]

    # compute the size of each of the first n-1 blocks
    block_size = np.ceil(length / float(n))

    # the indices at which the splits will occur
    ix = np.arange(block_size, length, block_size)

    return np.split(arr, ix, axis)

@torch.no_grad()
def predict(args):

    model = PedalNet.load_from_checkpoint(args.model)
    model.eval()
    
    train_data = pickle.load(open(args.train_data, "rb"))

    mean, std = train_data["mean"], train_data["std"]

    in_data, in_rate = librosa.load(args.input, sr=44100)
    
    assert in_rate == 44100, f"input data needs to be 44.1 kHz, current sampling rate is {in_rate}"

    sample_size = int(in_rate * args.sample_time)
    length = len(in_data) - len(in_data) % sample_size

    # split into samples
    in_data = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    # standardize
    in_data = (in_data - mean) / std

    # pad each sample with previous sample
    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    pred = []
    batches = math.ceil(pad_in_data.shape[0] / args.batch_size)
    for x in tqdm(np.array_split(pad_in_data, batches)):
        pred.append(model(torch.from_numpy(x)).numpy())

    pred = np.concatenate(pred)
    pred = pred[:, :, -in_data.shape[2] :]

    save(args.output, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/pedalnet.ckpt")
    parser.add_argument("--train_data", default="data.pickle")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    predict(args)