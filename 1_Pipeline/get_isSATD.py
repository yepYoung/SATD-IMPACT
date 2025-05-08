from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import recall_score, precision_score
import csv
import os
import logging
import random

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import tokenization
from modeling_multitask import BertConfig, BertForSequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
torch.cuda.set_device(0)


class InputExample(object):
    def __init__(self, text_a, label=None, dataset_label=None):
        self.text_a = text_a
        self.label = label
        self.dataset_label = dataset_label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, dataset_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dataset_label_id = dataset_label_id


class DataProcessor(object):
    def get_predict_example(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AllProcessor(DataProcessor):

    def get_predict_example(self, source_text, task):
        return self.create_example(source_text, "predict", task)

    def get_labels(self):
        return [["no", "yes"]]

    def create_example(self, source_text, set_type, task):
        
        # text = process.getdata(text)
        
        text_a = tokenization.convert_to_unicode(source_text)

        example = InputExample(text_a=text_a, label=None, dataset_label=task)
        return example


def convert_example_to_feature(example, label_list, max_seq_length, tokenizer):
    label_map_1 = {}
    for (i, label) in enumerate(label_list[0]):
        label_map_1[label] = i


    tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)


    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = None

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            dataset_label_id=int(example.dataset_label))

    return feature


def get_isSATD_with_0_or_1(text, task):
    processor = AllProcessor()
    label_list = processor.get_labels()
    device = torch.device("cuda")


    tokenizer = tokenization.FullTokenizer(
        vocab_file="Pipeline/Pipeline_tools/bert_config/vocab.txt", do_lower_case=True)

    bert_config = BertConfig.from_json_file("Pipeline/Pipeline_tools/bert_config/config.json")

    model = BertForSequenceClassification(bert_config, len(label_list))

    checkpoint = torch.load('pytorch_model16.bin', map_location=device)
    model.bert.load_state_dict(checkpoint['bert'])
    model.classifier1_merge.load_state_dict(checkpoint['classifier1_merge'])
    model.classifier2_merge.load_state_dict(checkpoint['classifier2_merge'])
    model.classifier3_merge.load_state_dict(checkpoint['classifier3_merge'])
    model.classifier4_merge.load_state_dict(checkpoint['classifier4_merge'])

    model.to(device)

    #================enter=================
    source_text = text
    task = task
    predict_example = processor.get_predict_example(source_text, task)
    predict_feature = convert_example_to_feature(predict_example, label_list, 128, tokenizer)
    input_ids = torch.tensor(predict_feature.input_ids, dtype=torch.long)
    input_ids = input_ids.reshape(1, input_ids.shape[0])

    input_mask = torch.tensor(predict_feature.input_mask, dtype=torch.long)
    input_mask = input_mask.reshape(1, input_mask.shape[0])

    segment_ids = torch.tensor(predict_feature.segment_ids, dtype=torch.long)
    segment_ids = segment_ids.reshape(1, segment_ids.shape[0])

    label_ids = torch.tensor(2, dtype=torch.long)

    dataset_label_id = torch.tensor(predict_feature.dataset_label_id, dtype=torch.long)
    id = dataset_label_id.item()
    list_id = [[id]]
    dataset_label_id = torch.tensor(list_id)


    model.eval()

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_id = None
        
        
    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, label_id, dataset_label_id)
        logits = logits.detach().cpu().numpy()
        softmaxs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        outputs = np.argmax(logits, axis=1)
        print(outputs[0])

    return outputs[0]
