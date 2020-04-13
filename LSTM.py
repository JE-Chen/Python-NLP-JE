# -*- coding: utf-8 -*-
'''
暫時停工
'''


import keras
import numpy as np
import random
import sys
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop

#載入訓練用文檔
path=keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt'
)
text=open(path,encoding='utf-8').read().lower()
print('Corpus Length : ',len(text))
'''
maxlen=60代表建構句子長度為60

step=3代表擷取句子時跳過3個單字

用set提取的每個單字集合按照編碼由小到大
'''
max_len=60
step=3
sentences=[]
next_chars=[]

for i in range(0,len(text)-max_len,step):
    sentences.append(text[i:i+max_len])
    next_chars.append(text[i+max_len])
print('Number of sentences : ',len(sentences))

chars=sorted(list(set(text)))
print('Unique characters : ',len(chars))

