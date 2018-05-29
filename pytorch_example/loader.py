import torch.utils.data as data

import os
import os.path
import torch

import librosa
import numpy as np

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]

CATEGORIES = """
Ba Bm Eg
""".split()

def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def find_classes(dir):
    #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes = CATEGORIES
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in CATEGORIES:
            continue
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects

def spect_loader(path):
    X, sample_rate = librosa.load(path, sr=None)
    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0)
    # print mfccs.shape
    mfccs = np.resize(mfccs, (1, mfccs.shape[0]))
    mfccs = np.resize(mfccs, (1, mfccs.shape[0],mfccs.shape[1]))
    # print mfccs.shape
    mfccs = torch.FloatTensor(mfccs)

    # print("mfccs dim ", mfccs.shape)
    # print(spect)
    return mfccs



class GCommandLoader(data.Dataset):

    def __init__(self, root, normalize=True, max_len=101):
        print root
        classes, class_to_idx = find_classes(root)
        print classes
        spects = make_dataset(root, class_to_idx)
        print spects
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = spect_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path)

        return spect, target

    def __len__(self):
        return len(self.spects)
