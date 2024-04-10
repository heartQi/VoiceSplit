import os
import random
import pandas
import librosa
random.seed(0)
all_speakers = []
output_dir = "../datasets/LibriSpeech2/"

vctk_dir = '/Users/mervin.qi/Desktop/PSE/Dataset/voicesplit_normalize/'
vctk_wavs_dir = 'dev-clean/'
sample_list = []

sample_rate = 16000
audio_len= int(sample_rate * 3) #time for 3 seconds

for root, dirs, files in os.walk(vctk_dir+vctk_wavs_dir):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        wav_files = [file for file in os.listdir(dir_path) if file.endswith('.wav')]
        if len(wav_files) > 0:
            all_speakers.append(dir)

for i in range(len(all_speakers)):
    speaker_clean = all_speakers[i].replace(' ', '')
    for j in range(i+1,len(all_speakers[i:])):
            interference_speaker = all_speakers[j].replace(' ', '')
            wav_samples = os.listdir(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean))
            
            clean_wav = random.choice(wav_samples)
            while librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,clean_wav), sr=sample_rate)[0].shape[0] < audio_len:
                clean_wav = random.choice(wav_samples)
            clean_wav = clean_wav.replace(speaker_clean, '')

            emb_wav = random.choice(wav_samples) # select one emb reference diferente then clean_wav
            while clean_wav == emb_wav and librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,emb_wav), sr=sample_rate)[0].shape[0] < audio_len: # its necessary for emb and clean not is same sample
                emb_wav = random.choice(wav_samples)
            emb_wav = emb_wav.replace(speaker_clean, '')

            # get samples interference samples
            wav_samples = os.listdir(os.path.join(vctk_dir,vctk_wavs_dir,interference_speaker))
            interference_wav = random.choice(wav_samples)
            while clean_wav == interference_wav and librosa.load(os.path.join(vctk_dir,vctk_wavs_dir,speaker_clean,interference_wav), sr=sample_rate)[0].shape[0] < audio_len: # its necessary for clean interference not is same text, its necessary because the texts in vctk is parallel
                interference_wav = random.choice(wav_samples)
            interference_wav = interference_wav.replace(interference_speaker, '')
            print(clean_wav,emb_wav,interference_wav)

            clean_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+clean_wav)
            emb_ref = os.path.join(vctk_wavs_dir, speaker_clean, speaker_clean+emb_wav)
            interference_ref = os.path.join(vctk_wavs_dir, interference_speaker, interference_speaker+interference_wav) 
            
            sample_list.append([clean_ref, emb_ref, interference_ref])


df = pandas.DataFrame(data=sample_list, columns=['clean_utterance','embedding_utterance','interference_utterance'])
df.to_csv(os.path.join(output_dir, "test_data_csv.csv"), index=False)




    
