#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from tqdm import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import time

# In[2]:


import pandas as pd
df = pd.read_csv('politeness.tsv', sep='\t')


# In[3]:


#df.head()


# In[4]:


#df['style'].value_counts()


# In[5]:


possible_labels = df['style'].unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    # label_dict[possible_label] = index
    label_dict[possible_label] = int(possible_label.split('_')[1])
label_dict


# In[6]:


df['label'] = df['style'].replace(label_dict)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[10]:


encoded_data_train = tokenizer.batch_encode_plus(
    df[df['split']=='train'].txt.values, 
    add_special_tokens=True, 
    return_attention_mask=True,
    max_length=256,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df['split']=='val'].txt.values, 
    add_special_tokens=True, 
    return_attention_mask=True,
    max_length=256,
    truncation=True, 
    padding='max_length', 
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    df[df['split']=='test'].txt.values, 
    add_special_tokens=True, 
    return_attention_mask=True,
    max_length=256,
    truncation=True, 
    padding='max_length', 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df['split']=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df['split']=='val'].label.values)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(df[df['split']=='test'].label.values)

# In[11]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


# In[12]:


#len(dataset_train), len(dataset_val)


# In[13]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)


# In[14]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 3

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)


# In[15]:


from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)


# In[16]:


epochs = 15

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


# In[17]:


from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

# In[18]:


import numpy as np
import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[19]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#print(device)


# In[20]:


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# In[ ]:

best_val_loss = float('inf')
early_stop_cnt = 0

#for epoch in tqdm(range(1, epochs+1)):
#    
#    model.train()
#
#    start_time = time.time()
#    
#    loss_train_total = 0
#
#    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
#    for batch in progress_bar:
#
#        model.zero_grad()
#        
#        batch = tuple(b.to(device) for b in batch)
#        
#        inputs = {'input_ids':      batch[0],
#                  'attention_mask': batch[1],
#                  'labels':         batch[2],
#                 }       
#
#        outputs = model(**inputs)
#        
#        loss = outputs[0]
#        loss_train_total += loss.item()
#        loss.backward()
#
#        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#        optimizer.step()
#        scheduler.step()
#        
#        #progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
#         
#        
#    #tqdm.write(f'\nEpoch {epoch}')
#    
#    loss_train_avg = loss_train_total/len(dataloader_train)            
#    #tqdm.write(f'Training loss: {loss_train_avg}')
#    
#    val_loss, predictions, true_vals = evaluate(dataloader_validation)
#
#    if val_loss < best_val_loss:
#        early_stop_cnt = 0
#
#    elif val_loss >= best_val_loss:
#        early_stop_cnt += 1
#
#    if(val_loss<best_val_loss):
#        best_val_loss = val_loss
#        torch.save(model.state_dict(), f'finetuned_BERT_best.model')
#    
#    val_f1 = f1_score_func(predictions, true_vals)
#    #tqdm.write(f'Validation loss: {val_loss}')
#    #tqdm.write(f'F1 Score (Weighted): {val_f1}')
#
#    end_time = time.time()
#    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#    
#    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)
#    print(f'\tTrain Loss: {loss_train_avg:.3f}', flush=True)
#    print(f'\t Val. Loss: {val_loss:.3f}', flush=True)
#
#    if early_stop_cnt == 3:
#        print('Early Stoping...', flush=True)
#        break


# In[ ]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)


# In[ ]:


model.load_state_dict(torch.load('finetuned_BERT_best.model', map_location=torch.device('cpu')))


# In[ ]:


_, predictions, true_vals = evaluate(dataloader_test)


# In[ ]:


accuracy_per_class(predictions, true_vals)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

lbls = [0,1,2,3,4,5,6,7,8,9]
classes = ['P_0','P_1','P_2', 'P_3', 'P_4', 'P_5', 'P_6','P_7','P_8','P_9']

preds_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_vals.flatten()

confusion = confusion_matrix(labels_flat, preds_flat, labels=lbls)
print('Confusion Matrix')
print(confusion)

#confusion_classes_tuples = list(zip(classes, confusion))
confusion_df = pd.DataFrame(confusion,
                     index = classes,
                     columns = classes)
print(confusion_df)
confusion_df.to_csv('confusion_matrix', sep='\t', encoding='utf-8')

print('\nClassification Report\n')
print(classification_report(labels_flat, preds_flat, target_names=classes))





