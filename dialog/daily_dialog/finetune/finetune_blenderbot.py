#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension


# In[2]:


#from ipywidgets import FloatProgress


# In[2]:


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[17]:


def read_data(data_path):
    src = []
    trg = []
    with open(data_path) as f:
        train_data = f.readlines()
        #print(len(train_data))
        for line in train_data:
            utterences = line.split('__eou__')
            del utterences[-1] #last item is '\n'
            src_utterence = []
            #trg_utterence = ''
            #for idx in range(0,(len(utterences)-1),2):
            for idx in range(len(utterences)-1):
                # src_utterence=src_utterence+'</s>'+trg_utterence
                
                src_utterence.append(utterences[idx])
                # if idx>0:
                #     src_utterence.append(trg_utterence)
                if(len(src_utterence)>5):
                   src_utterence = src_utterence[-5:]
                
                src.append('</s>'.join(src_utterence))
                trg_utterence = utterences[idx+1]
                trg.append(trg_utterence)
                
        return src,trg
base_path = '../../../dialog_data/dailydialog/'          
train_src, train_trg = read_data(base_path + 'train/dialogues_train.txt')
dev_src, dev_trg = read_data(base_path + 'validation/dialogues_validation.txt')
test_src, test_trg = read_data(base_path + 'test/dialogues_test.txt')


# In[ ]:


# train_src = train_src[0:2]
# train_trg = train_trg[0:2]
# dev_src = dev_src[0:2]
# dev_trg = dev_trg[0:2]


# In[27]:


with open(base_path + 'train/dialogues_train.txt') as myfile:
    head = [next(myfile) for x in range(1)]
print(head)

length = len(head[0].split('__eou__'))

print('\n')
for idx in range(length):
    print(train_src[idx])
    print(train_trg[idx])
    print('\n')


# In[28]:


import pandas as pd

train_df = pd.DataFrame(
    {'src': train_src,
     'trg': train_trg
    })
train_df.head()

dev_df = pd.DataFrame(
    {'src': dev_src,
     'trg': dev_trg
    })
dev_df.head()

test_df = pd.DataFrame(
    {'src': test_src,
     'trg': test_trg
    })
test_df.head()


# In[31]:


# model_name = "90MBB_facebook/blenderbot_small-90M/checkpoint-87000/"
# model_name = "facebook/blenderbot_small-90M"
# model_name = 'facebook/blenderbot-400M-distill/checkpoint-87176'
model_name = 'facebook/blenderbot-400M-distill'


# In[10]:


from tqdm import tqdm
# from transformers import BlenderbotSmallTokenizer
from transformers import BlenderbotTokenizer
import torch

tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
#tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)

train_src_encodings = tokenizer(train_df['src'].values.tolist(), truncation=True, padding=True, max_length=128)
train_trg_encodings = tokenizer(train_df['trg'].values.tolist(), truncation=True, padding=True, max_length=128)

dev_src_encodings = tokenizer(dev_df['src'].values.tolist(), truncation=True, padding=True, max_length=128)
dev_trg_encodings = tokenizer(dev_df['trg'].values.tolist(), truncation=True, padding=True, max_length=128)


# In[11]:


for key, value in train_src_encodings.items():
    print(key, len(value))


# In[12]:


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        # item['decoder_input_ids'] = torch.tensor(self.labels['input_ids'][idx])
        # item['decoder_attention_mask'] = torch.tensor(self.labels['attention_mask'][idx])

        return item

    def __len__(self):
        return len(self.labels['input_ids'])


# In[13]:


train_dataset = CreateDataset(train_src_encodings, train_trg_encodings)
dev_dataset = CreateDataset(dev_src_encodings, dev_trg_encodings)


# In[14]:


#train_dataset[0]


# In[15]:


#len(train_dataset)


# In[ ]:


# import datasets
# metric = datasets.load_metric("rouge")


# In[ ]:


# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # Rouge expects a newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     # Extract a few results
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
    
#     return {k: round(v, 4) for k, v in result.items()}


# In[16]:


from transformers import BlenderbotForConditionalGeneration
# from transformers import BlenderbotSmallForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
#model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name).to(device)

batch_size = 8
args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy = 'epoch',
    load_best_model_at_end=True,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=True
)


# In[17]:


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[18]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
    #compute_metrics=compute_metrics
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.evaluate()


# In[19]:


for idx in range(50):
    print('src:', test_df['src'].values.tolist()[idx])
    print('trg:', test_df['trg'].values.tolist()[idx])

    test_src = tokenizer(test_df['src'].values.tolist()[idx], truncation=True, padding=True, max_length=128, return_tensors='pt')
    generated_ids = model.generate(test_src["input_ids"].cuda())

    print('pred:', tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
    print('\n')


# In[ ]:




