import os
import random
import torch
import torch.utils.data
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram

from utils import load_filepaths_and_label


class MelDataset_filelist(torch.utils.data.Dataset):
    """
        1) loads audio and label pairs
        2) computes mel-spectrograms from audio files.
    """
    def __init__(self, filelist, hparams):
        self.filelist = load_filepaths_and_label(filelist)

        self.sample_rate = hparams.sample_rate
        self.transform_to_mel = MelSpectrogram(
            sample_rate=hparams.sample_rate,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            n_mels=hparams.n_mels,
            normalized=hparams.mel_normalized
        )

        random.seed(hparams.seed)
        random.shuffle(self.filelist)

    def get_mel_label_pair(self, audiopath_text_and_spk):
        # separate filename and text
        audiopath, label = audiopath_text_and_spk
        mel = self.get_mel(audiopath)
        return (mel, label)

    def get_mel(self, filename):
        audio, sampling_rate = torchaudio.load(filename, normalized=True)
        if sampling_rate != self.sample_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.sample_rate))
        mel = self.transform_to_mel(audio)
        mel = mel.squeeze(0)

        return mel

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.filelist[index])

    def __len__(self):
        return len(self.filelist)

class MelDataset_SpeechCommand(SPEECHCOMMANDS):
    def __init__(self, subset, hparams):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


        self.sample_rate = hparams.sample_rate
        self.transform_to_mel = MelSpectrogram(
            sample_rate=hparams.sample_rate,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            n_mels=hparams.n_mels,
            normalized=hparams.mel_normalized
        )


    def __getitem__(self, n: int):
        audio, sample_rate, label, _, _ =  super().__getitem__(n)
        if sample_rate != self.sample_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(sample_rate, self.sample_rate))
        mel = self.transform_to_mel(audio)
        mel = mel.squeeze(0)

        return (mel, label)


class MelLabelCollate():
    def __init__(self, n_frame_per_utt, labels):
        self.n_frame_per_utt = n_frame_per_utt
        self.labels = labels

    def label_to_index(self, word):
        # Return the position of the word in labels
        return torch.tensor(self.labels.index(word))


    def index_to_label(self, index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return self.labels[index]
        
    def __call__(self, batch):
        num_mels = batch[0][0].size(0)
        mels = torch.FloatTensor(len(batch), num_mels, self.n_frame_per_utt)
        label_ids = torch.LongTensor(len(batch))
        for i, x in enumerate(batch):
            mel, label = x
            label_ids[i] = self.label_to_index(label)
            mel_length = mel.size(1)
            if mel_length < self.n_frame_per_utt:
                mel = torch.cat([mel.tile((1, self.n_frame_per_utt//mel_length)), \
                        mel[:, :self.n_frame_per_utt%mel_length]], dim=1)
            elif mel_length > self.n_frame_per_utt:
                start = random.choice(range(0, mel_length-self.n_frame_per_utt))
                mel = mel[:, start:start+self.n_frame_per_utt]
            mels[i] = mel
            
        return mels, label_ids
