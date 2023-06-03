#!/usr/bin/env python
# coding: utf-8

# # Fine Tuning Transformer for Summary Generation

# In[1]:


get_ipython().system('pip install transformers -q')
get_ipython().system('pip install wandb -q')


# In[2]:


path = 'D:/jupyter_notebook/thesis/'


# # Ai허브에서 가져온 데이터를 이용해 데이터 전처리!!

# In[3]:


import numpy as np
import pandas as pd
import json


# In[4]:


# json 파일 가져오기

with open(path + '논문요약_0206_0.json', 'r', encoding='utf-8') as f:
  json_thesis_0206_0 = json.load(f)

with open(path + '논문요약_0206_1.json', 'r', encoding='utf-8') as f:
  json_thesis_0206_1 = json.load(f)

with open(path + '논문요약_0206_2.json', 'r', encoding='utf-8') as f:
  json_thesis_0206_2 = json.load(f)

with open(path + '논문요약_0220_0.json', 'r', encoding='utf-8') as f:
  json_thesis_0220_0 = json.load(f)

with open(path + '논문요약_0225_5_1.json', 'r', encoding='utf-8') as f:
  json_thesis_0225_5_1 = json.load(f)

with open(path + '논문요약_0225_7_0.json', 'r', encoding='utf-8') as f:
  json_thesis_0225_7_0 = json.load(f)


# In[5]:


# json 파일을 DataFrame 형식으로 바꾸기. 자연어 추후 전처리 필요

df_thesis_0206_0 = pd.DataFrame(json_thesis_0206_0['data'])
df_thesis_0206_1 = pd.DataFrame(json_thesis_0206_1['data'])
df_thesis_0206_2 = pd.DataFrame(json_thesis_0206_2['data'])
df_thesis_0220_0 = pd.DataFrame(json_thesis_0220_0['data'])
df_thesis_0225_5_1 = pd.DataFrame(json_thesis_0225_5_1['data'])
df_thesis_0225_7_0 = pd.DataFrame(json_thesis_0225_7_0['data'])


# In[6]:


df_thesis_0206_0


# In[7]:


thesis_0206_0 = list()
thesis_0206_1 = list()
thesis_0206_2 = list()
thesis_0220_0 = list()
thesis_0225_5_1 = list()
thesis_0225_7_0 = list()

for e in json_thesis_0206_0['data']:
  thesis_0206_0.append(e['summary_entire'][0])

for e in json_thesis_0206_1['data']:
  thesis_0206_1.append(e['summary_entire'][0])

for e in json_thesis_0206_2['data']:
  thesis_0206_2.append(e['summary_entire'][0])

for e in json_thesis_0220_0['data']:
  thesis_0220_0.append(e['summary_entire'][0])

for e in json_thesis_0225_5_1['data']:
  thesis_0225_5_1.append(e['summary_entire'][0])

for e in json_thesis_0225_7_0['data']:
  thesis_0225_7_0.append(e['summary_entire'][0])

df_thesis_0206_0 = pd.DataFrame(thesis_0206_0)
df_thesis_0206_1 = pd.DataFrame(thesis_0206_1)
df_thesis_0206_2 = pd.DataFrame(thesis_0206_2)
df_thesis_0220_0 = pd.DataFrame(thesis_0220_0)
df_thesis_0225_5_1 = pd.DataFrame(thesis_0225_5_1)
df_thesis_0225_7_0 = pd.DataFrame(thesis_0225_7_0)

#모든 dataframe을 concat으로 합쳐서 df라는 하나의 dataframe으로 만듦
df = pd.concat([df_thesis_0206_0, df_thesis_0206_1, df_thesis_0206_2, df_thesis_0220_0, df_thesis_0225_5_1, df_thesis_0225_7_0])


# In[8]:


#orginal_text 길이는 700이상 1000이하로 제한
#summary_text 길이는 150이상 200이하로 제한
#이유
#1. 데이터 길이 분포가 전처리 전보다 고르게 분포하도록 하기 위함
#2. 입력데이터들의 길이가 비슷한것이 모델의 학습효과에도 좋기 때문

df = df[
    (df['orginal_text'].str.len() <= 1000) &
    (df['orginal_text'].str.len() >= 700)
]

df = df[
    (df['summary_text'].str.len() <= 200) &
    (df['summary_text'].str.len() >= 150)
]
df


# In[9]:


#입력데이터의 길이 분포를 확인하기 위한 코드

import pandas as pd

sentences = df["orginal_text"] #이 때 나오는 그래프는 orginal_text의 길이 분포, 만약 summary_text의 분포를 확인하고 싶다면 summary_text로 바꿔서 길이 분포 확인할 수 있음

import matplotlib.pyplot as plt

min_len = 999
max_len = 0
sum_len = 0

for s in sentences:
  if len(s) < min_len :
    min_len = len(s)
  if len(s) > max_len :
    max_len = len(s)
  sum_len += len(s)

print('Min length : ', min_len)
print('Max length : ', max_len)
print('Average length : ', sum_len // len(sentences))

sen_length_cnt = [0] * max_len
for sen in sentences:
  sen_length_cnt[len(sen)-1] += 1

plt.bar(range(max_len), sen_length_cnt, width=1.0)
plt.show()


# In[10]:


import pandas as pd

#열이름 바꿔줌

df.rename(columns={'orginal_text': 'ctext'}, inplace=True)
df.rename(columns={'summary_text': 'text'}, inplace=True)

# DataFrame을 csv로 저장
df.to_csv('train.csv', index=True, encoding='utf-8')


# # 모델 만들기 시작!!!

# In[11]:


# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# WandB – Import the wandb library
import wandb


# In[12]:


# Checking out the GPU 
get_ipython().system('nvidia-smi')


# In[13]:


# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# In[14]:


# Login to wandb to log the model run and all the parameters
get_ipython().system('wandb login f2d7cd4fedd2822e534395005a236ac7b01e0036')


# ## 데이터셋 클래스 설정!!

# In[15]:


# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')
        
        # 토큰화된 텐서에서 모델에서 제공하는 두개의 키로 분류
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        
        # 딕셔너리 형태로 반환
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# ## 훈련함수 정의!!!

# In[16]:


# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network 

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()# 모델 훈련모드로 설정
    for _,data in enumerate(loader, 0): # enumerate함수를 사용하여 loader를 반복
        y = data['target_ids'].to(device, dtype = torch.long) # Tensor객체의 메서드인 to() ,장치이동, 데이터타입변경에 쓰임
        y_ids = y[:, :-1].contiguous() # Tensor 의 메모리 레이아웃을 연속적으로 만듬
        labels = y[:, 1:].clone().detach()# 새로운 텐서 labels를 만듬
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # 패딩 토큰 위치를 -100으로 설정하고 손실 함수가 이 위치들에 대한 예측 무시하도록 하기위함
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        
        if _%10 == 0:# 10번째 인덱스마다 loss출력
            wandb.log({"Training Loss": loss.item()})

        if _%500==0: # 500번째 인덱스마다
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            with torch.no_grad(): # 500번째마다 추론단계를 거치므로 grad 계산 불필요..!추론은 grad계산 필요없으니까!
                generated_ids = model.generate( # 텍스트 생성 함수. 트랜스포머 클래스안에 속해있는 함수이다.
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=300,  # Adjust the maximum length as needed
                    num_beams=2,  # Adjust the number of beams as needed
                    repetition_penalty=2.5,  # Adjust the repetition penalty as needed
                    early_stopping=True
                )
                # 리스트 컴프리헨션
                # tokenizer.decode 는 huggingFace 의 Tokenizer 클래스에 정의된 메서드
                # decode 메서드는 토큰ID 의 리스트를 원래의 텍스트로 디코딩하는 기능을 수행.
                # skip_special_tokens = True 옵션은 특수 토큰 디코딩 결과에서 제거
                decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
                print("Decoded Sentences:")
                print(decoded_sentences)
        
        optimizer.zero_grad() # 모든 모델 파라미터 그래디언트 0으로 설정, 이전에 계산된 그래디언트 제거하기 위해
        loss.backward() # 오류 역전파
        optimizer.step()# 모델 파라미터 업데이트
        # 이 세가지 단계는 각 배치에서 한번씩 수행된다!!


# ## 평가함수 정의!!

# In[17]:


def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            print("preds: ", preds)
            print("target: ", target)
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# In[18]:


# SentencePiece는 Google에서 개발한 오픈 소스 기반의 비지도 학습 토크나이저 및 텍스트 인코더로, 텍스트를 토큰으로 분리하는 데 사용
get_ipython().system('pip install sentencepiece')


# In[ ]:


from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import PreTrainedTokenizerFast, BartModel

def main():
    # WandB 초기화
    wandb.init(project="paper_summary_FINAL")

    # config 변수 설정, wandb.config 
    # 하이퍼파라미터 설정
    config = wandb.config          # Initialize config
    config.TRAIN_BATCH_SIZE = 1    # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 1    # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 5        # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 5 
    config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    config.SEED = 42               # random seed (default: 42)
    config.MAX_LEN = 1024
    config.SUMMARY_LEN = 200

    # 실험결과 재현위한 시드설정
    torch.manual_seed(config.SEED) # pytorch random seed
    np.random.seed(config.SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # 모델명 "KETI-AIR/ke-t5-base"
    model_name = "paust/pko-t5-base"

    # 설정한 사전 훈련 모델을 이용해 토크나이저 로드
    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    
    # 데이터파일 읽고, 요약텍스트(text), 원문(ctexct) 선택해서 데이터프레임으로 변형
    df = pd.read_csv(path+'train.csv',encoding='utf-8')
    df = df[['text','ctext']]
    df.ctext = 'summarize: ' + df.ctext
    print(df.head())

    
    # 학습데이터셋(80%)과 검증데이터셋 분할
    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state = config.SEED)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))


    # 위에서 정의한 CustomDataset을 사용해서 학습세트와 검증세트 생성
    # CustomDataset은 텍스트를 토큰화하고 모델에 입력할 수 있는 형식으로 변환하는 역할
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # 아래 만들어질 데이터로더에 들어갈 하이퍼 파라미터 생성
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    # DataLoader(training_set, **train_params)는 DataLoader(training_set, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)와 동일한 효과
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    
    
    # 모델 정의, 위에서 정의한 "paust/pko-t5-base" 를 사용한다.
    # to함수 사용하여 model gpu device로 이동시켜주기
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    # T5ForConditionalGeneration 모델의 인스턴스를 만들 때, 이 모델은 nn.Module의 모든 메서드와 속성을 상속받는다. 
    # model.parameters()는 모델의 모든 학습 가능한 매개변수를 Adam 옵티마이저에 전달.
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')
    
    # @@@@@훈련루프@@@@@
    # for문을 이용하여 훈련루프를 설정한다. config.TRAIN_EPOCHS로 에폭 수 설정!!
    # 실질적으로 훈련시작하는 구문!!
    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    # @@@@@검증루프@@@@@
    # saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    for epoch in range(config.VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv(path+'resultFINAL.csv')
        print('Output Files generated for review')

if __name__ == '__main__':
    main()

