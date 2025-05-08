import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import re, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import load_dataset,Dataset,DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader


HIDDEN_SIZE = 2048
MAX_LEN = 1024
NUM_LABELS = 9
SEED = 1
BATCH_SIZE = 1
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'
LOCAL_FILES_ONLY = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Class
class CustomModel(nn.Module):
    def __init__(self,checkpoint,num_labels, seed):
        super(CustomModel,self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True, local_files_only=LOCAL_FILES_ONLY))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # print(input_ids.shape)
        outputs = self.model(decoder_input_ids=input_ids, input_ids=input_ids, attention_mask=attention_mask)
        encoder_last_hidden_state = outputs.last_hidden_state
        sequence_output = self.dropout(encoder_last_hidden_state)
        # print(sequence_output.shape)
        sequence_view = sequence_output[:,0,:].view(-1, HIDDEN_SIZE)
        # print(sequence_view.shape)
        logits = self.classifier(sequence_view)
        # print(logits.shape)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # print(logits.view(-1, self.num_labels).shape, labels.view(-1).shape)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)
    

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
    
    
def create_data_loader(dataset, tokenizer, batch_size, max_length, isShuffle):
    dataset = CustomDataset(dataset, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=isShuffle)


def get_test_set(test_path):

    df_test = pd.read_csv(test_path)

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
    
    df_test['class'] = df_test['class'].map(label_mapping)

    dataset = {}
    
    for source in sorted(set(df_test['source'])):
        test_df  = df_test[df_test['source'] == source]      
        test_df = test_df[['source', 'text', 'class']]
        data = DatasetDict({"test": Dataset.from_pandas(test_df)})        
        data=data.rename_column("class","label")
        data=data.remove_columns(['source','__index_level_0__' ])
        dataset[source] = data

    return dataset


def test(checkpoint, test_path):
    
    test_results = {}
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, local_files_only=LOCAL_FILES_ONLY)
    tokenizer.model_max_len = MAX_LEN
    model = CustomModel(checkpoint=checkpoint, num_labels=NUM_LABELS, seed=SEED).to(device).eval()
    
    test_set = get_test_set(test_path)
    
    for source, data in test_set.items():
        
        test_results[source] = []

        test_dataloader = create_data_loader(data['test'], tokenizer, BATCH_SIZE, MAX_LEN, False)
        for batch in test_dataloader:
            
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                test_results[source] += batch_predictions.tolist()
                print(batch_predictions.tolist())
            
    return test_results



if __name__ == '__main__':
    
    checkpoint = 'train-flant5-XXL/checkpoint'
    test_path = 'train-flant5-XXL/test_all.csv'
    save_path = 'train-flant5-XXL/test_results'
    
    test_results = test(checkpoint, test_path)
    
    with open(os.path.join(save_path, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
            
