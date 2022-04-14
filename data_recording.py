import os
import random
from collections import defaultdict

import pyaudio
import librosa
import numpy as np
import soundfile as sf

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

out_dir = "recordings"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

labels_dict = defaultdict(list)

while(True):
    UID = input("Please enter the UID (Or enter FINSIH!! to finish the registration): ").strip()
    if UID == "FINISH!!":
        print("Finish the keyword registration process process.")
        break

    UID = UID.replace(" ", "_")
    UID = UID.upper()

    print("start recording utterance for {}... The recording will end in {} seconds".format(UID, RECORD_SECONDS))
    frames = []
    utterance = []
    keep_listening = True
    for i in range(RATE * RECORD_SECONDS // CHUNK):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
        array1 = np.frombuffer(data, dtype=np.float32)
        utterance.append(array1)

    print("End recording utterance for {}".format(UID))
    utterance = np.concatenate(utterance, axis=0)

    intervals = librosa.effects.split(utterance, top_db=20)
    count = 0
    for interval in intervals:
        start, end = interval
        if (end-start) < 0.3*RATE:
            continue
        else:
            segment = utterance[start:end]
            labels_dict[UID].append(segment)


total_list = []
for key in labels_dict.keys():
    for index, segment in enumerate(labels_dict[key]):
        wav_path = os.path.join(out_dir, "{}_{}.wav".format(key, index))
        sf.write(wav_path, segment, samplerate=RATE)
        total_list.append(wav_path+","+key+"\n")

random.shuffle(total_list)

num_of_train_data = int(len(total_list)*0.9)
train_list = total_list[:num_of_train_data]
valid_list = total_list[num_of_train_data:]

with open(os.path.join(out_dir, "train_filelist.txt"), "w") as f:
    for line in train_list:
        f.write(line)
    
with open(os.path.join(out_dir, "valid_filelist.txt"), "w") as f:
    for line in valid_list:
        f.write(line)


print("Please make sure you have all the wav segment file in directory {}, as well as two text file named train_filelist.txt & valid_filelist.txt ")
