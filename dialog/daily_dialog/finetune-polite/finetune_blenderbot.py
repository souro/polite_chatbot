#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension


# In[ ]:


#from ipywidgets import FloatProgress


# In[1]:


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# In[7]:


base_path = '../../../dialog_data/dailydialog/'

def read_data(data_path):
    src = []
    trg = []
    with open(base_path+data_path) as f:
        train_data = f.readlines() #f.read().splitlines()
        #print(len(train_data))
        for line in train_data:
            utterences = line.split('__eou__')
            del utterences[-1] #last item is '\n'
            src_utterence = []
            for idx in range(len(utterences)-1):
                # src_utterence=src_utterence+'</s>'+trg_utterence
                src_utterence.append(utterences[idx])
                
                if(len(src_utterence)>5):
                   src_utterence = src_utterence[-5:]
                
                src.append('</s>'.join(src_utterence))
                trg_utterence = utterences[idx+1].strip()
                #trg_utterence = trg_polite_dict[utterences[idx+1]]
                trg.append(trg_utterence)
                
        return src,trg
train_src, train_trg = read_data('train/dialogues_train.txt')
print(len(train_src), len(train_trg))
dev_src, dev_trg = read_data('validation/dialogues_validation.txt')
print(len(dev_src), len(dev_trg))
test_src, test_trg = read_data('test/dialogues_test.txt')
print(len(test_src), len(test_trg))


# In[4]:


import pickle


# In[ ]:


# res_path = 'daily_dialog/responses/'
# with open(res_path+'train_res.pkl', 'wb') as f:
#     pickle.dump(train_trg, f)
# with open(res_path+'dev_res.pkl', 'wb') as f:
#     pickle.dump(dev_trg, f)
# with open(res_path+'test_res.pkl', 'wb') as f:
#     pickle.dump(test_trg, f)


# In[8]:


train_trg_polite_responses = list()
# with open('../responses/train_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/train_polite_res.pkl', 'rb') as f:
    train_trg_polite_responses = pickle.load(f)
print(len(train_trg_polite_responses))

dev_trg_polite_responses = list()
# with open('../responses/dev_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/dev_polite_res.pkl', 'rb') as f:
    dev_trg_polite_responses = pickle.load(f)
print(len(dev_trg_polite_responses))

test_trg_polite_responses = list()
# with open('../responses/test_polite_res_direct.pkl', 'rb') as f:
with open('../responses/tag-gen/test_polite_res.pkl', 'rb') as f:
    test_trg_polite_responses = pickle.load(f)
print(len(test_trg_polite_responses))


# In[10]:


with open(base_path + 'train/dialogues_train.txt') as myfile:
    head = [next(myfile) for x in range(1)]
print(head)

length = len(head[0].split('__eou__'))

print('\n')
for idx in range(length):
    print('Context')
    print(10*'-')
    print(train_src[idx])
    
    print('\nUtterance')
    print(10*'-')
    print(train_trg[idx])
    
    print('\nPolite Utterance')
    print(10*'-')
    print(train_trg_polite_responses[idx])
    
    print(50*'=')
    print('\n')


# In[12]:


import pandas as pd

train_df = pd.DataFrame(
    {'contexts': train_src,
     'utterances': train_trg,
     'polite_utterances': train_trg_polite_responses
    })
train_df.head()

dev_df = pd.DataFrame(
    {'contexts': dev_src,
     'utterances': dev_trg,
     'polite_utterances': dev_trg_polite_responses
    })
dev_df.head()

test_df = pd.DataFrame(
    {'contexts': test_src,
     'utterances': test_trg,
     'polite_utterances': test_trg_polite_responses
    })
test_df.head()


# In[13]:


# model_name = "90MBB_facebook/blenderbot_small-90M/checkpoint-87000/"
# model_name = "facebook/blenderbot_small-90M"
# model_name = 'facebook/blenderbot-400M-distill/checkpoint-87176'
#model_name = 'facebook/blenderbot-400M-distill/checkpoint-28521'
model_name = 'facebook/blenderbot-400M-distill'


# In[ ]:


from tqdm.notebook import tqdm
# from transformers import BlenderbotSmallTokenizer
from transformers import BlenderbotTokenizer
import torch

tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
#tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)

train_src_encodings = tokenizer(train_df['contexts'].values.tolist(), truncation=True, padding=True, max_length=128)
train_trg_encodings = tokenizer(train_df['polite_utterances'].values.tolist(), truncation=True, padding=True, max_length=128)

dev_src_encodings = tokenizer(dev_df['contexts'].values.tolist(), truncation=True, padding=True, max_length=128)
dev_trg_encodings = tokenizer(dev_df['polite_utterances'].values.tolist(), truncation=True, padding=True, max_length=128)


# In[ ]:


for key, value in train_src_encodings.items():
    print(key, len(value))


# In[ ]:


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


# In[ ]:


train_dataset = CreateDataset(train_src_encodings, train_trg_encodings)
dev_dataset = CreateDataset(dev_src_encodings, dev_trg_encodings)


# In[ ]:


train_dataset[0]


# In[ ]:


len(train_dataset)


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


# In[14]:


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
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True
)


# In[15]:


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[16]:


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


# In[18]:


def gen(src):
    src_tknz = tokenizer(src, truncation=True, padding=True, max_length=128, return_tensors='pt')
    generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=128)

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# In[ ]:


for idx in range(5):
    print('src:', test_df['contexts'].values.tolist()[idx])
    print('trg:', test_df['utterances'].values.tolist()[idx])

    print('pred:', gen(test_df['contexts'].values.tolist()[idx]))
    print('\n')


# In[20]:


import pickle


# In[23]:


polite_dlg_responses_pred_direct = []
# for idx in range(len(test_df['src'].values.tolist())):
for idx in range(1000):
    polite_dlg_responses_pred_direct.append((gen(test_df['contexts'].values.tolist()[idx])))

print(len(polite_dlg_responses_pred_direct))

with open('Blenderbot_polite_pred_tag-gen.pkl', 'wb') as f:
    pickle.dump(polite_dlg_responses_pred_direct, f)


# In[ ]:



