import librosa
import numpy as np
from scipy import stats
import pandas as pd
import glob
import utils
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pickle
import math
import numpy as np
from nltk.tree import *
from nltk.stem import WordNetLemmatizer 
import random
import numpy as num
from math import exp
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer 


def tables():
    name='mfcc'
    size=20
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for m in moments:
        col=((name,m,'{:02d}'.format(i+1)) for i in range(size))
        columns.extend(col)
        
    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)
    return columns.sort_values()

files=glob.glob("./songs_file/*") 
track_ids=[]
songs_path=[]
for f in range(0,len(files)):
    track_ids.append(f)
    songs_path.append(files[f])

def compute(ids,song_dir):
    features = pd.Series(index=tables(), dtype=np.float32, name=ids)
    print(song_dir)
    y, sr = librosa.load(song_dir)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(y)/512) <= stft.shape[1] <= np.ceil(len(y)/512)+1
    

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    
    features["mfcc", 'mean'] = np.mean(f, axis=1)
    features["mfcc", 'std'] = np.std(f, axis=1)
    features["mfcc", 'skew'] = stats.skew(f, axis=1)
    features["mfcc", 'kurtosis'] = stats.kurtosis(f, axis=1)
    features["mfcc", 'median'] = np.median(f, axis=1)
    features["mfcc", 'min'] = np.min(f, axis=1)
    features["mfcc", 'max'] = np.max(f, axis=1)
    return features
   
import speech_recognition as sr 
  
import os 
  
from pydub import AudioSegment 
from pydub.silence import split_on_silence 
import pickle

dbfile = open('tag_vector', 'rb')      
lyr = pickle.load(dbfile) 
dbfile.close()

lyr=lyr['top']


token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+|\$+|\&+|\#+|\*+|\++|;+', gaps = True)
lem = WordNetLemmatizer() 

feature={'audio':[],'lyric':[]}
for i in range(0,len(track_ids)):
    feature['audio'].append(compute(track_ids[i],songs_path[i]))
    
    lyric=''
    
    with sr.AudioFile(songs_path[i]) as source:
        audio_data = sr.record(source)
        text = sr.recognize_google(audio_data)
        lyric+=text
    
    lyric=lyric.lower()
    lyric=token(lyric)
    lyric=[lem.lemmatize(k) for k in lyric]
    
    arr=[0 for i in range(5000)]
    
    for i in lyric:
        
        for j in range(1,len(lyr)):
            
            if i==lyr[j]:
                
                arr[i]+=1
                
    feature['lyric'].append(arr)
    
print("Index table is created")
index = open('query_vector', 'ab')
pickle.dump(feature, index)                
index.close()
print("Index table is stored")
