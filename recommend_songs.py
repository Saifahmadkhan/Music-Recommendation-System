import numpy as np
from scipy import stats
import pandas as pd
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
from numpy import dot
from numpy.linalg import norm

dbfile = open('query_vector', 'rb')      
query = pickle.load(dbfile) 
dbfile.close()

dbfile = open('tag_vector', 'rb')      
song = pickle.load(dbfile) 
dbfile.close()

wt=0.5

K=25 ##########################_NO_OF_SONG_RETURN_##################################

topk=0
map=0
token = RegexpTokenizer('\s+|\-+|\.+|\@+|\t+|\n+|[0-9]+|\"+|\>+|\,+|\?+|\:+|\{+|\(+|\[+|\)+|\}+|\]+|\<+|\_+|\!+|\/+|\|+|\\+|\*+|\=+|\^+|\$+|\&+|\#+|\*+|\++|;+', gaps = True)
for i in range(len(query['audio'])):
    
    arr=np.array(query['audio'][i])*(1-wt)+wt*np.array(query['lyric'][i])
    
    cos_sim=[]
    m=0
    print("Recommanded song are- \n")
    for train_id in range(len(song['final'])):
        a=song['final'][train_id]
        cos_sim.append([dot(a,arr)/(norm(a)*norm(arr)),train_id])
        
    cos_sim.sort(reverse=True)
    for i1 in range(K):
        print ("\t",cos_sim[i1][1])
    feedback=input("Give feedback by typing space seperated index (atleast relevant song) ")
    feedback=feedback.split(' ')
    if len(feedback)>1:
        feedback=[int(i) for i in feedback]
        topk+=len(feedback)
        cou=0
        for i in feedback:
            cou+=1.0
            m+=cou/(i+1)
        
    if len(feedback)!=0:
        map+=m/len(feedback)
    print()
    print()

if len(query['audio'])!=0:
    print("Top K value- ",topk/len(query['audio']))
    print("MAP value- ",map/len(query['audio']))
    
