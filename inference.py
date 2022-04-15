import os
import pickle
import argparse

import torch
import librosa
import pyaudio
import numpy as np

from torchaudio.transforms import MelSpectrogram

from x_vector import X_vector
from hparams import create_hparams
from utils import to_device

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"

def inference(model, stream, device, labels):

    transform_to_mel = MelSpectrogram(
            sample_rate=hparams.sample_rate,
            n_fft=hparams.n_fft,
            win_length=hparams.win_length,
            hop_length=hparams.hop_length,
            f_min=hparams.f_min,
            f_max=hparams.f_max,
            n_mels=hparams.n_mels,
            normalized=hparams.mel_normalized
        )

    while(True):
        _ = input("Please press enter to start recording!").strip()
        print("start recording utterance... The recording will end in {} seconds".format(RECORD_SECONDS))
        frames = []
        utterance = []

        for i in range(RATE * RECORD_SECONDS // CHUNK):
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.append(data)
            array1 = np.frombuffer(data, dtype=np.float32)
            utterance.append(array1)

        print("End recording utterance. Start recongnizing the input utterence")
        utterance = np.concatenate(utterance, axis=0)
        utterance, _ = librosa.effects.trim(utterance, top_db=20)

        utterance = torch.from_numpy(utterance)
        
        ## Normalize the utterance
        utterance = utterance / max(abs(utterance))
        mels = transform_to_mel(utterance)
        mels = mels.unsqueeze(0)
        
        mels = to_device(mels, device).float()
        predict = model(mels).argmax(dim=1).item()
        print(labels[predict])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, default="output_dir",
                        help='directory to save checkpoints')

    args = parser.parse_args()

    hparams = create_hparams()
    checkpoint_path = os.path.join(args.output_directory, "best_checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use {} as training device".format(device))

    ### Load label
    with open(os.path.join(args.output_directory, "labels.pt"), "rb") as infile:
        labels = pickle.load(infile)
        labels = sorted(labels)

    ### Load model
    model = X_vector(hparams, n_labels=len(labels)).to(device)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model.eval()

    ### objects for recording
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    inference(model, stream, device, labels)