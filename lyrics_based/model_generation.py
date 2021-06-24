import os
import codecs
import string
import os
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
from math import exp
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import WordNetLemmatizer 
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def read(file):
	fp = codecs.open(file,"r",encoding='utf-8', errors='ignore')
	text = fp.read()
	return text

dbfile = open('final_dataset', 'rb')      
song = pickle.load(dbfile) 
dbfile.close()

lt=[]
xt=[]
yt=[]

l1=[]
x1=[]
y1=[]

l2=[]
x2=[]
y2=[]

tag={'Pop_Rock':0,'Rap':0 ,'Country':0,'RnB':0,'Latin':0,'Electronic':0,'Religious':0} #'International'}
# 'Folk'
cou=0
tt=[]
for k in tag.keys():
    
    for i in range(len(song[k])):
        
        arr=[0 for j in range(9)]
        arr[cou]=1
        #tt.append([song[k][i][0],song[k][i][1],arr])
        tt.append([song[k][i][0],song[k][i][1],cou])
        
    
    cou+=1

random.shuffle(tt)

for i in tqdm(range(len(tt))):
    
    xt.append(tt[i][1])
    lt.append(tt[i][0])
    yt.append(tt[i][2])
    

part=0.8

x1=xt[:int(len(xt)*part)]
y1=yt[:int(len(yt)*part)]
l1=lt[:int(len(xt)*part)]


x2=xt[int(len(xt)*part):]
y2=yt[int(len(yt)*part):]
l2=lt[int(len(xt)*part):]


from sklearn.neural_network import MLPClassifier
clf3 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(50,150,80),random_state=1,max_iter=500)
clf2=MLPClassifier(alpha=1e-5, hidden_layer_sizes=(200,400,150), random_state=1) #   ,early_stopping=True)
from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(random_state=0,n_estimators=50) 

from sklearn.svm import SVC
clf5 =SVC(kernel='rbf')



import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(base_estimator=clf1,n_estimators=30,random_state=0)



#x1,y1=SMOTE().fit_resample(x1, y1)

print("Starting... ")
clf.fit(x1,y1)

o=clf.predict(x2)
print("End... ")
pred_aud = clf.predict_proba(x2)

cou=0
tol=0

pos=[0 for i in range(len(pred_aud[0]))]
pos1=[0 for i in range(len(pred_aud[0]))]
pos2=[0 for i in range(len(pred_aud[0]))]

for i in tqdm(range(len(o))):
    
    
    tol+=1.0
    
    pos1[y2[i]]+=1
    if o[i]==y2[i]:
        
        pos[o[i]]+=1
        cou+=1.0
        

print(cou/tol*100)


#l1,y1=SMOTE().fit_resample(l1, y1)


print("Starting... ")
clf.fit(l1,y1)

o=clf.predict(l2)
print("End... ")


pred_ly = clf.predict_proba(l2)

cou=0.0
tol=0.0
for i in tqdm(range(len(o))):
    
    tol+=1.0
    if o[i]==y2[i]:
        
        cou+=1.0
        pos2[o[i]]+=1
        

print(cou/tol*100)

print()
print()
label=[k for k in tag.keys()]
print(label)
print()
print(pos)
print(pos1)
print(list(np.array(pos)/np.array(pos1)*100))
print()
print(pos2)
print(pos1)
print(list(np.array(pos)/np.array(pos1)*100))
print()
print()

x_axis=[]
y_axis=[]

for i in range(11):
    
    alpha=i/10
    
    acc=0
    x_axis.append(alpha)
    for j in range(len(pred_ly)):
        
        a=alpha*(np.array(pred_ly[j]))+(1-alpha)*(np.array(pred_aud[j]))
        
        a=list(a)
        
        m=0
        ind=0
        
        for k in range(len(pred_aud[0])):
            
            if a[k]>m:
                
                m=a[k]
                ind=k
        
        if ind==y2[j]:
            
            acc+=1.0
    
    print("weight ",alpha," Acc- ",acc/tol*100)
    y_axis.append(acc/tol*100)
    
plt.plot(x_axis,y_axis, label = "Accuracy") 
plt.scatter(x_axis,y_axis,color='r')
plt.ylabel('Accuracy ->') 
plt.xlabel('Lyrics Weight ->') 
plt.title('Accuracy vs Weight') 
plt.legend() 
plt.show()


vector={'final':[],'audio':[],'lyric':[]}

wt=0.5

for i in range(len(y_axis)):
    
    if y_axis[i]>m:
        
        m=y_axis[i]
        wt=x_axis[i]

print("Storing Audio weight... ")
clf.fit(xt,yt)

audiov=clf.predict_proba(xt)


import pickle

'''
file='model_audio_weight.sav'
pickle.dump(clf,open(file,'wb'))
print()
print('Audio Model is stored')
print()
'''

print("Storing Lyrics... ")
clf.fit(lt,yt)

lyricv=clf.predict_proba(lt)

finalv=[]

for i in range(len(lyricv)):
    
    s=(1-wt)*(audiov[i])+(wt)*(lyricv[i])
    
    finalv.append(s)
    
audiov=list(audiov)
lyricv=list(lyricv)

vector['audio']=audiov
vector['lyric']=lyricv
vector['final']=finalv


path='Million song\\mxm_dataset_train.txt'
text=read(path)
text=text.split('\n')
word=text[0].split(',')[1:]
vector['top']=word
text=text[1:len(text)-1]



file='tag_vector'
pickle.dump(vector,open(file,'wb'))

'''
file='model_lyric_weight.sav'
pickle.dump(clf,open(file,'wb'))
print()
'''
print('Lyric Model is stored')
print()
