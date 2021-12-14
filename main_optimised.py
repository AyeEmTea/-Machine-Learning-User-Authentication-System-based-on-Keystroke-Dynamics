###################################### Importing General libraries ##############################################
import sys
from functools import lru_cache, cmp_to_key
from heapq import merge, heapify, heappop, heappush
from math import *
from collections import defaultdict as dd, deque, Counter as C
from itertools import combinations as comb, permutations as perm
from bisect import bisect_left as bl, bisect_right as br, bisect
from time import perf_counter
from fractions import Fraction
import copy
import time
###################################### Importing General libraries ##############################################







###################################### Importing DS libraries ####################################################
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split



###################################### Importing DS libraries ####################################################







####################################### Utility functions #######################################################
starttime = time.time()
mod = int(pow(10, 9) + 7)
mod2 = 998244353
def data(): return sys.stdin.readline().strip()
def out(*var, end="\n"): sys.stdout.write(' '.join(map(str, var))+end)
def L(): return list(sp())
def sl(): return list(ssp())
def sp(): return map(int, data().split())
def ssp(): return map(str, data().split())
def l1d(n, val=0): return [val for i in range(n)]
def l2d(n, m, val=0): return [l1d(n, val) for j in range(m)]

try:
    # sys.setrecursionlimit(int(pow(10,6)))
    sys.stdin = open("input.txt", "r")
    #sys.stdout = open("result.txt", "w")
except:
    pass
#sys.stdout = open("results.txt", "w")
####################################### Utility functions #######################################################





###################################### Data preprocessing #########################################################
def make_dict():
    D={}
    for i in range(0,26):
        D[chr(ord("A")+i)]=[]
        
    for i in range(0,26):
        for j in range(0,26):
            x=chr(ord("A")+i)
            y=chr(ord("A")+j)
            D[x+y]=[]
    return D
reg_users=["17EC35002","17EC35004","17EC35035","17EC10008","17EC10026","17EC10063","17EC35032","17EC35045","IMPOSTER_1","IMPOSTER_2"]
data=make_dict()
data['user']=[]
for p1 in reg_users:
    temp=make_dict()
    mxlen=0
    path=p1+"/Train Data/hold_time/"
    for i in range(26):
        x=chr(ord("A")+i)
        try:
            y=open(path+x+".txt",'r')
            sm=0
            cnt=0
            for l in y:
                z=float(l)
                sm+=z
                cnt+=1
                temp[x].append(sm/cnt)
        except:
            pass
        mxlen=max(mxlen,len(temp[x]))
    path=p1+"/Train Data/latencies/"
    for i in range(26):
        for j in range(26):
            x=chr(ord("A")+i)+chr(ord("A")+j)
            try:
                y=open(path+x+".txt",'r')
                sm=0
                cnt=0
                for l in y:
                    z=float(l)
                    sm+=z
                    cnt+=1
                    temp[x].append(sm/cnt)
            except:
                pass
            mxlen=max(mxlen,len(temp[x]))
    for k in temp:
        while(len(temp[k])!=mxlen):
            if len(temp[k])==0:
                temp[k].append(0.00001)
            else:
                temp[k].append(temp[k][-1])
        data[k]+=temp[k][::]
    for i in range(mxlen):
        data['user'].append(p1)
        
###################################### Data preprocessing #########################################################
        
        
        
        
        
        
##################################### Implementation of model #####################################################

data = pd.DataFrame(data)
data.drop_duplicates(inplace = True)
Y=data['user'] 
X=data.drop(['user'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

##################################### Implementation of model #####################################################
    
    
#################################### Training and Validation of model ##############################################
models=[]
Ai,Bi,Ci,Di,Ei=np.array_split(X_train,5)
ai,bi,ci,di,ei=np.array_split(y_train,5)
folds=["11110","11101","11011","10111","01111"]
Folds=[Ai,Bi,Ci,Di,Ei]
Tests=[ai,bi,ci,di,ei]
scores=[]
for f in folds:
    train_frames=[]
    results=[]
    cross=0
    for i in range(5):
        if f[i]=="1":
            train_frames.append(Folds[i])
            results.append(Tests[i])
        else:
            cross=i
    TRAIN=pd.concat(train_frames,sort=False)
    RESULT=pd.concat(results,sort=False)
    
    CV_X=Folds[cross][::]
    CV_Y=Tests[cross][::]
    
    gmm = GaussianMixture(n_components=len(reg_users))
    gmm.fit(TRAIN)
    U=list(RESULT)
    z=[]
    for usr in reg_users:
        z.append(U.count(usr))
    
    print("Training data count of users:")
    for i in range(len(z)):
        print(reg_users[i]," : ",z[i])
    print()
    labels = gmm.predict(TRAIN)
    frame = pd.DataFrame(TRAIN)
    frame['cluster'] = labels
    frame['user']=RESULT
    d={}
    for rol in reg_users:
        d[rol]=[0 for i in range(len(reg_users))]
    for i in list(frame.index.values):
        d[frame['user'][i]][frame['cluster'][i]]+=1
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1)
    print("No. of examples in each cluster for the users")
    print(pd.DataFrame(d))
    print()
    dummy={}
    for rol in d:
        
        d[rol]=d[rol].index(max(d[rol]))
        dummy[rol]=[d[rol]]
    print("Final clusters assigned")
    print(pd.DataFrame(dummy))
    
        
    labels = gmm.predict(CV_X)
    frame = pd.DataFrame(CV_X)
    frame['cluster'] = labels
    frame['user']=CV_Y
    res={}
    score=0
    total=0
    for i in list(frame.index.values):
        if frame['cluster'][i]==d[frame['user'][i]]:
            score+=1
        total+=1
    acc=score/total
    acc*=100
    print("Accuracy for the cross validation set", '%.5f'%acc, "%")
    models.append(gmm)

#gmm = GaussianMixture(n_components=len(reg_users))
#gmm.fit(X_train)
ACCS=[]
for gmm in models:
    labels = gmm.predict(X_train)
    frame = pd.DataFrame(X_train[::])
    frame['cluster'] = labels
    frame['user']=y_train
    d={}
    for rol in reg_users:
        d[rol]=[0 for i in range(len(reg_users))]
    for i in list(frame.index.values):
        d[frame['user'][i]][frame['cluster'][i]]+=1
    #print("Clusters", *range(len(reg_users)))
    for rol in d:
        #print(rol,*d[rol])
        d[rol]=d[rol].index(max(d[rol]))
        
    
    labels = gmm.predict(X_test)
    frame = pd.DataFrame(X_test[::])
    frame['cluster'] = labels
    frame['user']=y_test
    res={}
    score=0
    total=0
    for i in list(frame.index.values):
        if frame['cluster'][i]==d[frame['user'][i]]:
            score+=1
        total+=1
    acc=score/total
    acc*=100
    ACCS.append(acc)
acc=max(ACCS)
print("Accuracy for the Test set", '%.5f'%acc, "%")

    
    
        
        


#################################### Training and Validation of model #########################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    