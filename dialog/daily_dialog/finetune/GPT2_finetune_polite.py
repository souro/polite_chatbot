#!/usr/bin/env python
# coding: utf-8

# In[16]:


import time
import datetime

import seaborn as sns
import numpy as np
import random

# import matplotlib.pyplot as plt
# %matplotlib inline

import nltk
nltk.download('punkt')


# In[17]:


import pickle


# In[18]:


import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[19]:


import pickle


# In[20]:


train_utterances_polite = list()
# with open('../responses/train_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/train_polite_res.pkl', 'rb') as f:
    train_utterances_polite = pickle.load(f)
    train_utterances_polite = list(map(str.strip, train_utterances_polite))
print(len(train_utterances_polite))

dev_utterances_polite = list()
# with open('../responses/dev_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/dev_polite_res.pkl', 'rb') as f:
    dev_utterances_polite = pickle.load(f)
    dev_utterances_polite = list(map(str.strip, dev_utterances_polite))
print(len(dev_utterances_polite))

test_utterances_polite = list()
# with open('../responses/test_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/test_polite_res.pkl', 'rb') as f:
    test_utterances_polite = pickle.load(f)
    test_utterances_polite = list(map(str.strip, test_utterances_polite))
print(len(test_utterances_polite))


# In[21]:


import itertools

base_path = '../../../dialog_data/dailydialog/'

def read_data(data_path):
    contexts = []
    utterances = []
    #inputs = []
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
                #inputs.append(ip + turns[idx+1].strip())
                
        return contexts, utterances #, inputs
train_contexts, train_utterances = read_data('train/dialogues_train.txt')
dev_contexts, dev_utterances = read_data('validation/dialogues_validation.txt')
test_contexts, test_utterances = read_data('test/dialogues_test.txt')


# In[22]:


with open(base_path + 'test/dialogues_test.txt') as myfile:
    head = [next(myfile) for x in range(1)]
print(head)

length = len(head[0].split('__eou__'))

print('\n')
for idx in range(int(length)):
    print('Context')
    print(10*'-')
    print(dev_contexts[idx])
    
    print('\nUtterance')
    print(10*'-')
    print(dev_utterances[idx])
    
    print('\nPolite Utterance')
    print(10*'-')
    print(test_utterances_polite[idx])
    
    # print('\nInput')
    # print(10*'-')
    # print(dev_inputs[idx])
    
    print(50*'=')
    print('\n')


# In[23]:


import pandas as pd

train_df = pd.DataFrame(
    {'contexts': train_contexts,
     'utterances': train_utterances,
     'polite_utterances': train_utterances_polite
    })
train_df.head()

dev_df = pd.DataFrame(
    {'contexts': dev_contexts,
     'utterances': dev_utterances,
     'polite_utterances': dev_utterances_polite
    })
dev_df.head()

test_df = pd.DataFrame(
    {'contexts': test_contexts,
     'utterances': test_utterances,
     'polite_utterances': test_utterances_polite
    })
test_df.head()


# In[24]:


from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup


# In[25]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', eos_token='<|endoftext|>', pad_token='<|pad|>')
tokenizer.add_special_tokens({"additional_special_tokens": ['<|bot|>', '<|user|>']})


# In[26]:


print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))


# In[27]:


batch_size = 3


# In[28]:


class GPT2Dataset(Dataset):
    
    def __init__(self, contexts, utterances, tokenizer, max_length=256):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.cntxt_ids = []

        for context, utterance in zip(contexts, utterances):

            input_encoding_dict = tokenizer(context + utterance + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(input_encoding_dict['input_ids']))
            self.attn_masks.append(torch.tensor(input_encoding_dict['attention_mask']))

            context_encoding_dict = tokenizer(context + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length") 
            self.cntxt_ids.append(torch.tensor(context_encoding_dict['input_ids']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.cntxt_ids[idx]


# In[29]:


train_dataset = GPT2Dataset(train_df['contexts'].values.tolist(), train_df['utterances'].values.tolist(), tokenizer, max_length=256)
dev_dataset = GPT2Dataset(dev_df['contexts'].values.tolist(), dev_df['utterances'].values.tolist(), tokenizer, max_length=256)

# To train with polite utterances, (Change #1)
# train_dataset = GPT2Dataset(train_df['contexts'].values.tolist(), train_df['polite_utterances'].values.tolist(), tokenizer, max_length=256)
# dev_dataset = GPT2Dataset(dev_df['contexts'].values.tolist(), dev_df['polite_utterances'].values.tolist(), tokenizer, max_length=256)


# In[30]:


# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
dev_dataloader = DataLoader(
            dev_dataset,
            sampler = SequentialSampler(dev_dataset), # Pull out batches sequentially.
            batch_size = batch_size
        )


# In[31]:


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


# In[32]:


epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8


# In[33]:


optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )


# In[34]:


# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)


# In[35]:


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


# In[37]:


total_t0 = time.time()

training_stats = []

model = model.to(device)

best_val_loss = float('inf')
early_stop_cnt = 0

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        #b_labels = batch[0].to(device)
        b_labels = torch.where(batch[0] != batch[2], batch[0], -100).to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss
        #print('Batch {:>5,}  of  {:>5,} Loss: {:>5,}'.format(step, len(train_dataloader), batch_loss))

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in dev_dataloader:
        
        b_input_ids = batch[0].to(device)
        #b_labels = batch[0].to(device)
        b_labels = torch.where(batch[0] != batch[2], batch[0], -100).to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids,
                             attention_mask = b_masks,
                             labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(dev_dataloader)
    
    if avg_val_loss < best_val_loss:
        early_stop_cnt = 0

    elif avg_val_loss >= best_val_loss:
        early_stop_cnt += 1

    if(avg_val_loss<best_val_loss):
        best_val_loss = avg_val_loss
        
        # Need to change the name when training with polite data, (Change #2)
        # model.save_pretrained('GPT2_best_model')
        # tokenizer.save_pretrained('GPT2_best_model')
    
    validation_time = format_time(time.time() - t0)    

    print("Validation Loss: {0:.2f}".format(avg_val_loss))
    print("Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )
    
    # Need to change the name when training with polite data
    # with open('GPT2_training_stats.pkl', 'wb') as f:
    #     pickle.dump(training_stats, f)
    
    if early_stop_cnt == 3:
        print('Early Stoping...', flush=True)
        break

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


# In[38]:


# # Display floats with two decimal places.
# pd.options.display.max_rows = 2

# # Create a DataFrame from our training statistics.
# df_stats = pd.DataFrame(data=training_stats)

# # Use the 'epoch' as the row index.
# df_stats = df_stats.set_index('epoch')

# # A hack to force the column headers to wrap.
# #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# # Display the table.
# df_stats


# In[39]:


# # Use plot styling from seaborn.
# sns.set(style='darkgrid')

# # Increase the plot size and font size.
# sns.set(font_scale=1.5)
# plt.rcParams["figure.figsize"] = (12,6)

# # Plot the learning curve.
# plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
# plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# # Label the plot.
# plt.title("Training & Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.xticks([1, 2, 3, 4])

# plt.show()


# In[40]:


model.eval()
def gen(context):
    context_input_ids = tokenizer.encode(context + '<|endoftext|>', truncation=True, max_length=256, padding="max_length", return_tensors='pt')
    gen_outputs = model.generate(context_input_ids.cuda(), max_length=300, min_length=25, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)
    return tokenizer.batch_decode(gen_outputs[:, context_input_ids.shape[-1]:], skip_special_tokens=True)[0]


# In[41]:


print(test_df['contexts'].values.tolist()[35])
op = gen(test_df['contexts'].values.tolist()[35])
print(op)


# In[42]:


import pickle


# In[43]:


dlg_responses_pred_wo = []
contexts = test_df['contexts'].values.tolist()
for idx in range(1000):
#for context in test_df['contexts'].values.tolist():
    context = contexts[idx]
    dlg_responses_pred_wo.append(gen(context))

print(len(dlg_responses_pred_wo))

# Need to change the name when training with polite data, (Change #3)
with open('GPT2_finetune_tag-gen_polite.pkl', 'wb') as f:
    pickle.dump(dlg_responses_pred_wo, f)


# In[ ]:





# In[ ]:



