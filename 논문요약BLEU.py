#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from nltk.translate.bleu_score import sentence_bleu


# In[6]:


# CSV 파일 로드
data = pd.read_csv('./result2.csv',encoding='ANSI') 

# 두 번째 열과 세 번째 열을 단어 단위로 분할
data.iloc[:, 1] = data.iloc[:, 1].apply(lambda x: x.split())  # summary
data.iloc[:, 2] = data.iloc[:, 2].apply(lambda x: x.split())  # target

# BLEU 점수 계산
data['bleu_score'] = data.apply(lambda row: sentence_bleu([row.iloc[2]], row.iloc[1]), axis=1)

# 평균 BLEU 점수 출력
average_bleu = data['bleu_score'].mean()
print("Average BLEU Score: ", average_bleu)


# In[ ]:




