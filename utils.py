import torch
import numpy as np
from scipy.io.wavfile import read
from sklearn.metrics import confusion_matrix, accuracy_score

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask



def load_filepaths_and_label(filename, split=","):
    with open(filename, encoding='utf-8') as f:
        filepaths_text_and_speaker = [line.strip().split(split) for line in f]
    return filepaths_text_and_speaker


def to_device(x, device):
    x = x.contiguous()
    x = x.to(device)
    return torch.autograd.Variable(x)


def get_acc_and_confusion_matrix(labels, predictions):
    acc = accuracy_score(labels, predictions)
    c_matrix = confusion_matrix(labels, predictions)
    
    return acc, c_matrix


