from re import S
from turtle import mode
import os
import json
from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import T5Model
import torch
import json
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
import time
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW,get_scheduler, AutoModelForSeq2SeqLM
import torch.nn.functional as F

#======参数===================
checkpoint = "checkpoint"
USE_LoRA = True
LOCAL_FILES_ONLY = True

HIDDEN_SIZE = 2048
MAX_LEN = 512
BATCH_SIZE = 6
LR=0.00005
num_epochs = 8
SEED = 1
ADD_CLASSIFICATION_LAYER = True

TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
NUM_LABELS = 9
lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
#============================
class CustomT5Model(T5Model):
    def forward(self, **kwargs):
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        return super().forward(**kwargs)


# Model Class
class CustomModel(nn.Module):
    
    def __init__(self, checkpoint, num_labels, seed):
        super(CustomModel,self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_labels = num_labels
        if USE_LoRA == False:
            self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY), device_map='auto')
        else:
            self.model = get_peft_model(CustomT5Model.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY), device_map='auto'), lora_config)
        
        self.dropout = nn.Dropout(0.1).to(self.model.device)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels).to(self.model.device)
    
    def get_device(self):
        return self.model.device    
        
    def forward(self, input_ids=None, attention_mask=None,labels=None):
        # print(input_ids.shape)
        outputs = self.model(decoder_input_ids=input_ids, input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_state = outputs.last_hidden_state
        sequence_output = self.dropout(encoder_last_hidden_state)
        # print(sequence_output.shape)
        sequence_view = sequence_output[:,0,:].view(-1,HIDDEN_SIZE)
        # print(sequence_view.shape)
        logits = self.classifier(sequence_view)
        # print(logits.shape)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(loss=loss, logits=logits)



# Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']

        encoded = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        
        return input_ids, attention_mask, label


#tools
def create_data_loader(dataset, tokenizer, batch_size, max_length, isShuffle):
    dataset = CustomDataset(dataset, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle)


def get_dataset(train_path, dev_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(dev_path)

    label_mapping = {
        "NON-SATD": 0,
        "Code": 1,
        "Test": 2,
        "Build": 3,
        "Architecture": 4,
        "Documentation": 5,
        "Defect": 6,
        "Design": 7,
        "Requirement": 8
    }
    df_train['class'] = df_train['class'].map(label_mapping)
    df_test['class'] = df_test['class'].map(label_mapping)



    dataset = {}

    # for source in sorted(set(df_train['source'])):
        
        
    # train_df = df_train[df_train['source'] == source] 
    # train_df = train_df[['source', 'text', 'class']]   

    df_train = df_train[['source', 'text', 'class']] 
    df_test = df_test[['source', 'text', 'class']] 
    
    # test_df  = df_test[df_test['source'] == source]
    # test_df = test_df[['source', 'text', 'class']]
    
    data = DatasetDict({"train": Dataset.from_pandas(df_train), "test": Dataset.from_pandas(df_test)})        
    data=data.rename_column("class","label")

    # data=data.remove_columns(['source', '__index_level_0__'])
    # data=data.remove_columns(['source'])
    
    # dataset[source] = data
    

    # return dataset    

    return data


def tokenize(batch):
    inputs = tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=MAX_LEN)
    if ADD_CLASSIFICATION_LAYER:
        return inputs
    




dataset = get_dataset('train_all_01satd.csv', 'dev_all_01satd.csv') 
# print(dataset)

# with open('@show.dataset.json', 'w') as f:
#     json.dump(dataset, f, indent=4)


tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
tokenizer.model_max_len = MAX_LEN

projects_pred = {}


# begin
# for source, data in dataset.items():
print('=======================================')
# print('------', source, '------')

torch.cuda.empty_cache()
gc.collect()

labels = [str(x) for x in list(set(dataset['train']['label']))]

model = CustomModel(checkpoint=checkpoint, num_labels=NUM_LABELS, seed=SEED)
optimizer = AdamW(model.parameters(), lr=LR)   
        
train_dataloader = create_data_loader(dataset['train'], tokenizer, BATCH_SIZE, MAX_LEN, True)
test_dataloader = create_data_loader(dataset['test'], tokenizer, BATCH_SIZE, MAX_LEN, False)

num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = int(0.1 * num_training_steps)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

losses = []
for epoch in range(num_epochs):
    
    model.train()
    projects_pred['results'] = []

    for batch in tqdm(train_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(model.get_device())
        attention_mask = attention_mask.to(model.get_device())
        labels = labels.to(model.get_device())
    
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss     
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print('epoch',epoch+1,'of',num_epochs)
    
    del batch
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

    model.eval()

    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(model.get_device())
        attention_mask = attention_mask.to(model.get_device())
        labels = labels.to(model.get_device())
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            print(batch_predictions)
            for item in batch_predictions:
                projects_pred['results'].append(item.item())
            
    model.train()
    
    save_path = f'@train_results/train_res_epoch{str(epoch+1)}'
    if not os.path.exists(save_path):    
        os.makedirs(save_path)


    if epoch == num_epochs - 1: 
    # torch.save(model.state_dict(),  os.path.join(save_path, 'pytorch_model.bin'))
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(save_path, 'pytorch_model.bin'),
            _use_new_zipfile_serialization=False  # 禁用新式压缩格式
        )

    with open(os.path.join(save_path, 'projects_pred.json'), 'w') as f:
        json.dump(projects_pred, f, indent=4)
    # projects_pred[source] = []