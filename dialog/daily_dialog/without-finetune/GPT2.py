#!/usr/bin/env python
# coding: utf-8

# In[321]:


import time
import datetime

#import seaborn as sns
import numpy as np
import random

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import nltk
nltk.download('punkt')


# In[322]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[323]:


import itertools

base_path = '../../../dialog_data/dailydialog/'

def read_data(data_path):
    contexts = []
    utterances = []
    inputs = []
    with open(base_path+data_path) as f:
        train_data = f.readlines()
        for line in train_data:
            turns = line.split('__eou__')
            del turns[-1] #last item is '\n'
            
            chat_history = []
            for idx in range(len(turns)-1):
                chat_history.append(turns[idx].strip())
                
                if(len(chat_history)>5):
                    chat_history = chat_history[-5:]
                
                if(len(chat_history)%2!=0):
                    who = itertools.cycle(['<|user|> ', '<|bot|> '])
                    ip = next(who)
                elif(len(chat_history)%2==0):
                    who = itertools.cycle(['<|bot|> ', '<|user|> '])
                    ip = next(who)
                for chat in chat_history:
                    ip = ip + chat + '\n' + next(who)
                contexts.append(ip)
                utterances.append(turns[idx+1].strip())
                inputs.append(ip + turns[idx+1].strip())
                
        return contexts, utterances, inputs
train_contexts, train_utterances, train_inputs = read_data('train/dialogues_train.txt')
dev_contexts, dev_utterances, dev_inputs = read_data('validation/dialogues_validation.txt')
test_contexts, test_utterances, test_inputs = read_data('test/dialogues_test.txt')


# In[324]:


with open(base_path + 'test/dialogues_test.txt') as myfile:
    head = [next(myfile) for x in range(1)]
print(head)

length = len(head[0].split('__eou__'))

print('\n')
for idx in range(int(length)):
    print('Context')
    print(10*'-')
    print(test_contexts[idx])
    
    print('\nUtterance')
    print(10*'-')
    print(test_utterances[idx])
    
    print('\nInput')
    print(10*'-')
    print(test_inputs[idx])
    print(50*'=')
    print('\n')


# In[325]:


import pandas as pd

train_contexts, train_utterances, train_inputs
train_df = pd.DataFrame(
    {'contexts': train_contexts,
     'utterances': train_utterances,
     'inputs': train_inputs
    })
train_df.head()

dev_df = pd.DataFrame(
    {'contexts': dev_contexts,
     'utterances': dev_utterances,
     'inputs': dev_inputs
    })
dev_df.head()

test_df = pd.DataFrame(
    {'contexts': test_contexts,
     'utterances': test_utterances,
     'inputs': test_inputs
    })
test_df.head()


# In[326]:


from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


# In[327]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
tokenizer.add_special_tokens({"additional_special_tokens": ['<|bot|>', '<|user|>']})


# In[328]:


print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))


# In[329]:


batch_size = 8


# In[333]:


configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda")
model.cuda()

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[210]:


def gen(context):
    context_input_ids = tokenizer.encode(context + '<|endoftext|>', truncation=True, max_length=256, padding="max_length", return_tensors='pt')
    gen_outputs = model.generate(context_input_ids.cuda(), max_length=300, min_length=25, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)
    return tokenizer.batch_decode(gen_outputs[:, context_input_ids.shape[-1]:], skip_special_tokens=True)[0]

# In[40]:


import pickle


# In[ ]:


dlg_responses_pred_wo = []
contexts = test_df['contexts'].values.tolist()
for idx in range(1000):
#for context in test_df['contexts'].values.tolist():
    context = contexts[idx]
    dlg_responses_pred_wo.append(gen(context))

print(len(dlg_responses_pred_wo))

with open('dlg_responses_wo_pred_lists_GPT2.pkl', 'wb') as f:
    pickle.dump(dlg_responses_pred_wo, f)


# In[ ]:




