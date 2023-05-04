#!/usr/bin/env python
# coding: utf-8

# In[1]:


#base_path = '../../tagger_data/as_it_is/'
base_path = '../../gen_data/as_it_is/'

with open(base_path + 'train.src') as f:
    train_src = f.readlines()
    print(len(train_src))
with open(base_path + 'train.trg') as f:
    train_trg = f.readlines()
    print(len(train_trg))
with open(base_path + 'dev.src') as f:
    dev_src = f.readlines()
    print(len(dev_src))
with open(base_path + 'dev.trg') as f:
    dev_trg = f.readlines()
    print(len(dev_trg))
with open(base_path + 'test.src') as f:
    test_src = f.readlines()
    print(len(test_src))
with open(base_path + 'test.trg') as f:
    test_trg = f.readlines()
    print(len(test_trg))


# In[2]:


import pandas as pd

train_df = pd.DataFrame(
    {'src': train_src,
     'trg': train_trg
    })
#train_df.head()

dev_df = pd.DataFrame(
    {'src': dev_src,
     'trg': dev_trg
    })
#dev_df.head()

test_df = pd.DataFrame(
    {'src': test_src,
     'trg': test_trg
    })
#test_df.head()


# In[3]:


print(train_df['src'][1])
train_df.src = train_df.src.str.replace('\n', '')
dev_df.src = dev_df.src.str.replace('\n', '')
test_df.src = test_df.src.str.replace('\n', '')
print(train_df['src'][1])

print(train_df['trg'][1])
train_df.trg = train_df.trg.str.replace('\n', '')
dev_df.trg = dev_df.trg.str.replace('\n', '')
test_df.trg = test_df.trg.str.replace('\n', '')
print(train_df['trg'][1])

#test_df.head()


# In[4]:


train_df.src = train_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')
dev_df.src = dev_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')
test_df.src = test_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')

# train_df.trg = train_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')
# dev_df.trg = dev_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')
# test_df.trg = test_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '<mask>')

# train_df.src = train_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '')
# dev_df.src = dev_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '')
# test_df.src = test_df.src.str.replace('\[P_[a-zA-Z0-9_]+\]', '')

# train_df.trg = train_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '')
# dev_df.trg = dev_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '')
# test_df.trg = test_df.trg.str.replace('\[P_[a-zA-Z0-9_]+\]', '')

# train_df['src'][1]
#test_df.head()


# In[5]:


# import json

# def df_to_dict_file(df, file):
#     inp = df['src'].values.tolist()
#     oup = df['trg'].values.tolist()
#     dict_list = []
#     for idx in range(len(df.index)):
#         dict_list.append({'in':inp[idx], 'out':oup[idx]})
#     with open(file, "w") as outfile:
#         json.dump({'data':dict_list}, outfile)

# df_to_dict_file(train_df, 'json_data/train.json')
# df_to_dict_file(dev_df, 'json_data/dev.json')
# df_to_dict_file(test_df, 'json_data/test.json')


# In[6]:


# model_name = 'facebook/bart-base'
model_name = 'facebook/bart-base/checkpoint-106260'

# In[7]:


from transformers import BartTokenizer 
import torch


tokenizer = BartTokenizer.from_pretrained(model_name)

train_src_encodings = tokenizer(train_df['src'].values.tolist(), truncation=True, padding=True, max_length=128)
train_trg_encodings = tokenizer(train_df['trg'].values.tolist(), truncation=True, padding=True, max_length=128)

dev_src_encodings = tokenizer(dev_df['src'].values.tolist(), truncation=True, padding=True, max_length=128)
dev_trg_encodings = tokenizer(dev_df['trg'].values.tolist(), truncation=True, padding=True, max_length=128)


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


#train_dataset[0]


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


# In[ ]:


from transformers import BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = BartForConditionalGeneration.from_pretrained(model_name)

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


# In[ ]:


from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# In[ ]:


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


# trainer.train()


# In[ ]:


# trainer.evaluate()


# In[ ]:


# for idx in range(50):
#     print('src:', test_df['src'].values.tolist()[idx])
#     print('trg:', test_df['trg'].values.tolist()[idx])

#     test_src = tokenizer(test_df['src'].values.tolist()[idx], truncation=True, padding=True, max_length=128, return_tensors='pt')
#     generated_ids = model.generate(test_src["input_ids"].cuda())

#     print('pred:', tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
#     print('\n')


# # In[ ]:


# test_ips = ['send me the text files.', 'look into this issue.', 'good bye.', 'okay.', 'do it.', 'yes, go ahead and remove it.', 'jon please use this resignation letter in lieu of the one sent on friday']
# for test_ip in test_ips:
#     test_src = tokenizer(test_ip, truncation=True, padding=True, max_length=128, return_tensors='pt')
#     generated_ids = model.generate(test_src["input_ids"].cuda())
#     print('src:', test_ip)
#     print('pred:', tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
#     print('\n')

def gen(src):
    src_tknz = tokenizer(src, truncation=True, padding=True, max_length=128, return_tensors='pt')
    generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=128)

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
def polite_res(r_file,w_file):
    lines=None
    with open(r_file, 'r') as f:
        lines = f.readlines()
    with open(w_file, 'w') as f:
        for line in lines:
            f.write(f"{gen(line)}\n")
            # print(line)
            # print(gen(line))
            # print('\n')

res_path = '../../dialog/daily_dialog/responses/'
polite_res(res_path+'train_polite_res_mask', res_path+'train_polite_res_gen')
polite_res(res_path+'dev_polite_res_mask', res_path+'dev_polite_res_gen')
polite_res(res_path+'test_polite_res_mask', res_path+'test_polite_res_gen')