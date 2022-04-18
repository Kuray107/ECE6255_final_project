# ECE6255_final_project
This script is for ECE6255 final project: Spoken Utterance Recognition (SUR).
- In order to run our code, please first follow instructions in Stage 0. 
- If you only want to test our 5-UIDs (i,e., "Hello", "Good Morning", "Maybe", "Hey Siri", and "Oh") fine-tuned SUR system, please skip Stage 1 & 2 and run Stage 3 `python inference.py -o output_dir` directly. 
- If you would like to go through the whole process, please delete the directories `output_dir` and `recordings` and start from stage 1. 


## Stage 0: Environmental Setup
- Install python requirements. We recommand using conda for enviroment setup:
    1. Create a conda environment. `conda create --name ECE6255 python=3.8`
    2. Install [Pytorch ver. 1.10.1](https://pytorch.org/get-started/previous-versions/). you can run `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch` if your OS is XOS. Otherwise, please follow the link instructions to install the package.
    3. Install Pyaudio `conda install -c anaconda pyaudio`
    4. Install soundfile and librosa `conda install -c conda-forge pysoundfile librosa`
- Alternatively, run `conda env create -f environment.yml` if your OS is XOS.

## Stage 1: Recording
1. Run `python data_recording.py` 
2. After start running, the system will first ask you to enter the UID. After you enter the UID, the system will start recording for 10 seconds.
3. You can repeat the same keyword several times during the recording process (with a pause between each utterance). Note that your voice should be loud and clear.
4. You could also re-enter the same UID and add more utterances into training data if you would like to.
5. The system will keep asking you to enter UID until you enter FINISH!! to end the process.
6. The system will create a directory named `recordings`. Check if you have it before moving to the next stage.


## Stage 2: Training using a pre-trained model
Since the recording process usually cannot generate a lot of training utterance, we first pretrained our X-vector model on the [Google Speech Command dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). The pretrained weight is stored as `GSC_pretrained_model.pt`, so you don't need to worry about how to do model pretraining. 

1. Run `python finetune_on_recording.py -o <name of output directory>`
2. By default, the system will create a directory according to the input argument (default: output_dir), which stores model checkpoints. This process will keep running for a while, but you can stop it using Ctrl+C if the validation accuracy is high enough.  

## Stage 3: Utterance Recognition
1. Run `python inference.py -o <name of output directory>` (default: output_dir).
2. The system will ask you to start recording by press enter. You only need to say the utterance once this time. 
3. After the recording process ends, the system will output the UID of this utterance and ask you for another utterance.



