"""BERT finetuning runner."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.metrics import recall_score, precision_score

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pandas as pd

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import tokenization
from modeling_multitask import BertConfig, BertForSequenceClassification
from optimization import BERTAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# torch.cuda.set_device(1)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, dataset_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            dataset_label: 1:imdb 2:yelp p 3:yelp f 4:trec 5:yahoo 6:ag 7:dbpedia
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.dataset_label = dataset_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, dataset_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.dataset_label_id = dataset_label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AllProcessor(DataProcessor):
    """Processor for the all 7 data sets."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print(data_dir + "======================")
        train_data_issue = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                         # data_dir + "_issue_train.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'train_2issues_process.csv'), header=None, sep=",").values
        train_data_request = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                         # data_dir + "_pr_train.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'train_3pull-requests_process.csv'), header=None, sep=",").values
        train_data_commit = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        # data_dir + "_commit_train.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'train_4commit-messages_process.csv'), header=None, sep=",").values
        train_data_code_comments = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        # data_dir + "_code_comments_train.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'train_1code-comments_process.csv'), header=None, sep=",").values
        train_data_issue = np.delete(train_data_issue, 0, 0)
        train_data_request = np.delete(train_data_request, 0, 0)
        train_data_commit = np.delete(train_data_commit, 0, 0)
        train_data_code_comments = np.delete(train_data_code_comments, 0, 0)
        print(train_data_code_comments[0])

        return self._create_examples(train_data_issue, train_data_request, train_data_commit, train_data_code_comments,
                                     "train")

    def get_dev_examples(self, data_dir):
        test_data_issue = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        # data_dir + "_issue_test.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'dev_2issues_process.csv'), header=None, sep=",").values
        test_data_request = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        # data_dir + "_pr_test.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'dev_3pull-requests_process.csv'), header=None, sep=",").values
        test_data_commit = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        # data_dir + "_commit_test.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'dev_4commit-messages_process.csv'), header=None, sep=",").values
        test_data_code_comments = pd.read_csv(
            # os.path.join("satd/multi_train/" + data_dir + "_data/",
                        #  data_dir + "_code_comments_test.csv"), header=None, sep=",").values
            os.path.join(data_dir, 'dev_1code-comments_process.csv'), header=None, sep=",").values
        test_data_issue = np.delete(test_data_issue, 0, 0)
        test_data_request = np.delete(test_data_request, 0, 0)
        test_data_commit = np.delete(test_data_commit, 0, 0)
        test_data_code_comments = np.delete(test_data_code_comments, 0, 0)
        print("=====================")
        print(test_data_issue[0])

        return self._create_examples(test_data_issue, test_data_request, test_data_commit, test_data_code_comments,
                                     "test")

    def get_labels(self):
        """See base class."""
        return [["no", "yes"],
                ["no", "yes"],
                ["no", "yes"],
                ["no", "yes"]]

    def _create_examples(self, lines_issue, lines_request, lines_commit, lines_code_comments, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines_issue):
            # if i>147:break
            guid = "%s-%s" % (set_type, i)

            text_a = tokenization.convert_to_unicode(str(line[4]))
            label = tokenization.convert_to_unicode(str(line[7]))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="2"))

        for (i, line) in enumerate(lines_request):
            # if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[4]))
            label = tokenization.convert_to_unicode(str(line[7]))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="3"))

        for (i, line) in enumerate(lines_commit):
            # if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[4]))
            label = tokenization.convert_to_unicode(str(line[7]))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="4"))

        for (i, line) in enumerate(lines_code_comments):
            # if i>147:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[4]))
            label = tokenization.convert_to_unicode(str(line[7]))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, dataset_label="1"))
        print("create examples?")
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map_1 = {}
    for (i, label) in enumerate(label_list[0]):
        label_map_1[label] = i

    label_map_2 = {}
    for (i, label) in enumerate(label_list[1]):
        label_map_2[label] = i

    label_map_3 = {}
    for (i, label) in enumerate(label_list[2]):
        label_map_3[label] = i

    label_map_4 = {}
    for (i, label) in enumerate(label_list[3]):
        label_map_4[label] = i

    features_1 = []
    features_2 = []
    features_3 = []
    features_4 = []
    print("convert start.")
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.dataset_label == "1": label_id = label_map_1[example.label]
        if example.dataset_label == "2": label_id = label_map_2[example.label]
        if example.dataset_label == "3": label_id = label_map_3[example.label]
        if example.dataset_label == "4": label_id = label_map_4[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
        if example.dataset_label == "1":
            features_1.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "2":
            features_2.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "3":
            features_3.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))
        if example.dataset_label == "4":
            features_4.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    dataset_label_id=int(example.dataset_label)))

    return features_1, features_2, features_3, features_4


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default="train/bert-base-uncased/config.json",
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default="all",
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default="train/bert-base-uncased/vocab.txt",
                        type=str,
                        required=False,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default="train/bert-base-uncased/pytorch_model.bin",
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--discr",
                        default=False,
                        action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--patience',
                        type=int,
                        default=8,
                        help="early stopping")
    args = parser.parse_args()

    processors = {
        "all": AllProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 1
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    early_stopping_count = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)

    model = BertForSequenceClassification(bert_config, len(label_list))
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)

    #  torch.save(model.bert.state_dict(),"pytorch_model.bin")
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ['bias', 'gamma', 'beta']
    if args.discr:
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.']
        group2 = ['layer.6.', 'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.']
        group3 = ['layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                     'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate / 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': 0.01, 'lr': args.learning_rate * 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate / 1.5},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay_rate': 0.0,
             'lr': args.learning_rate * 1.5},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    optimizer = BERTAdam(optimizer_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features_1, eval_features_2, eval_features_3, eval_features_4 = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_1], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_1], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_1], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_1], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_1], dtype=torch.long)
    eval_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_1 = DataLoader(eval_data_1, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_2], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_2], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_2], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_2], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_2], dtype=torch.long)
    eval_data_2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_2 = DataLoader(eval_data_2, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_3], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_3], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_3], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_3], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_3], dtype=torch.long)
    eval_data_3 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_3 = DataLoader(eval_data_3, batch_size=args.eval_batch_size, shuffle=False)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features_4], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features_4], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features_4], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features_4], dtype=torch.long)
    all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in eval_features_4], dtype=torch.long)
    eval_data_4 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_dataset_label_ids)
    eval_dataloader_4 = DataLoader(eval_data_4, batch_size=args.eval_batch_size, shuffle=False)
    best_f1 = 0

    if args.do_train:
        train_features_1, train_features_2, train_features_3, train_features_4 = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_1], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_1], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_1], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_1], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_1], dtype=torch.long)
        train_data_1 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                     all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_1 = RandomSampler(train_data_1)
        else:
            train_sampler_1 = DistributedSampler(train_data_1)
        train_dataloader_1 = DataLoader(train_data_1, sampler=train_sampler_1, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_2], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_2], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_2], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_2], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_2], dtype=torch.long)
        train_data_2 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                     all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_2 = RandomSampler(train_data_2)
        else:
            train_sampler_2 = DistributedSampler(train_data_2)
        train_dataloader_2 = DataLoader(train_data_2, sampler=train_sampler_2, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_3], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_3], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_3], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_3], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_3], dtype=torch.long)
        train_data_3 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                     all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_3 = RandomSampler(train_data_3)
        else:
            train_sampler_3 = DistributedSampler(train_data_3)
        train_dataloader_3 = DataLoader(train_data_3, sampler=train_sampler_3, batch_size=args.train_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in train_features_4], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features_4], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features_4], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features_4], dtype=torch.long)
        all_dataset_label_ids = torch.tensor([f.dataset_label_id for f in train_features_4], dtype=torch.long)
        train_data_4 = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                     all_dataset_label_ids)
        if args.local_rank == -1:
            train_sampler_4 = RandomSampler(train_data_4)
        else:
            train_sampler_4 = DistributedSampler(train_data_4)
        train_dataloader_4 = DataLoader(train_data_4, sampler=train_sampler_4, batch_size=args.train_batch_size)

        print("len(train_features_1)=", len(train_features_1))
        print("len(train_features_2)=", len(train_features_2))
        print("len(train_features_3)=", len(train_features_3))
        print("len(train_features_4)=", len(train_features_4))
        a = []
        for i in range(int(len(train_features_1) / args.train_batch_size)):
            a.append(1)
        for i in range(int(len(train_features_2) / args.train_batch_size)):
            a.append(2)
        for i in range(int(len(train_features_3) / args.train_batch_size)):
            a.append(3)
        for i in range(int(len(train_features_4) / args.train_batch_size)):
            a.append(4)
        print("len(a)=", len(a))
        random.shuffle(a)
        print("a[:20]=", a[:20])

        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            random.shuffle(a)
            print("a[:20]=", a[:20])
            epoch += 1
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, number in enumerate((tqdm(a, desc="Iteration"))):
                if number == 1: batch = train_dataloader_1.__iter__().__next__()
                if number == 2: batch = train_dataloader_2.__iter__().__next__()
                if number == 3: batch = train_dataloader_3.__iter__().__next__()
                if number == 4: batch = train_dataloader_4.__iter__().__next__()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, dataset_label_id = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()  # We have accumulated enought gradients
                    model.zero_grad()
                    global_step += 1

            model.eval()
            # issue
            eval_loss, eval_accuracy, eval_precision_issue, eval_recall_issue, eval_f1_issue = 0, 0, 0, 0, 0
            predict_label_list_issue = []
            origin_label_list_issue = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_issue_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_2:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_issue.extend(label_ids)
                    predict_label_list_issue.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_issue = eval_loss
            eval_accuracy_issue = eval_accuracy
            eval_recall_issue = recall_score(origin_label_list_issue, predict_label_list_issue)
            eval_precision_issue = precision_score(origin_label_list_issue, predict_label_list_issue)
            eval_f1_issue = 2.0 * (eval_recall_issue * eval_precision_issue) / (
                    eval_recall_issue + eval_precision_issue)

            # request
            eval_loss, eval_accuracy, eval_precision_pr, eval_recall_pr, eval_f1_pr = 0, 0, 0, 0, 0
            predict_label_list_pr = []
            origin_label_list_pr = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_request_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_3:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_pr.extend(label_ids)
                    predict_label_list_pr.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_pr = eval_loss
            eval_accuracy_pr = eval_accuracy
            eval_recall_pr = recall_score(origin_label_list_pr, predict_label_list_pr)
            eval_precision_pr = precision_score(origin_label_list_pr, predict_label_list_pr)
            eval_f1_pr = 2.0 * (eval_recall_pr * eval_precision_pr) / (eval_recall_pr + eval_precision_pr)

            # commit
            eval_loss, eval_accuracy, eval_precision_commit, eval_recall_commit, eval_f1_commit = 0, 0, 0, 0, 0
            predict_label_list_commit = []
            origin_label_list_commit = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_commit_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_4:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_commit.extend(label_ids)
                    predict_label_list_commit.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_commit = eval_loss
            eval_accuracy_commit = eval_accuracy
            eval_recall_commit = recall_score(origin_label_list_commit, predict_label_list_commit)
            eval_precision_commit = precision_score(origin_label_list_commit, predict_label_list_commit)
            eval_f1_commit = 2.0 * (eval_recall_commit * eval_precision_commit) / (
                    eval_recall_commit + eval_precision_commit)
            print("len_test_commit===============" + str(len(predict_label_list_commit)))

            # code comments
            eval_loss, eval_accuracy, eval_precision_code_comments, eval_recall_code_comments, eval_f1_code_comments = 0, 0, 0, 0, 0
            predict_label_list_code_comments = []
            origin_label_list_code_comments = []
            nb_eval_steps, nb_eval_examples = 0, 0
            with open(os.path.join(args.output_dir, "results_code_comments_" + str(epoch) + ".txt"), "w") as f:
                for input_ids, input_mask, segment_ids, label_ids, dataset_label_id in eval_dataloader_1:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask, label_ids, dataset_label_id)

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output in outputs:
                        f.write(str(output) + "\n")
                    tmp_eval_accuracy = np.sum(outputs == label_ids)
                    origin_label_list_code_comments.extend(label_ids)
                    predict_label_list_code_comments.extend(outputs)
                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples
            eval_loss_code_comments = eval_loss
            eval_accuracy_code_comments = eval_accuracy
            eval_recall_code_comments = recall_score(origin_label_list_code_comments, predict_label_list_code_comments)
            eval_precision_code_comments = precision_score(origin_label_list_code_comments,
                                                           predict_label_list_code_comments)
            eval_f1_code_comments = 2.0 * (eval_recall_code_comments * eval_precision_code_comments) / (
                    eval_recall_code_comments + eval_precision_code_comments)

            avg_precision = (
                                    eval_precision_issue + eval_precision_pr + eval_precision_commit + eval_precision_code_comments) / 4.0
            avg_recall = (eval_recall_issue + eval_recall_pr + eval_recall_commit + eval_recall_code_comments) / 4.0
            avg_f1 = (eval_f1_issue + eval_f1_pr + eval_f1_commit + eval_f1_code_comments) / 4.0

            if avg_f1 > best_f1:
                torch.save({
                    'bert': model.bert.state_dict(),
                    'classifier1_0': model.classifier1_0.state_dict(),
                    'classifier1_1': model.classifier1_1.state_dict(),
                    'classifier1_2': model.classifier1_2.state_dict(),
                    'classifier1_3': model.classifier1_3.state_dict(),
                    'classifier1_4': model.classifier1_4.state_dict(),
                    'classifier1_5': model.classifier1_5.state_dict(),
                    'classifier1_6': model.classifier1_6.state_dict(),
                    'classifier1_7': model.classifier1_7.state_dict(),
                    'classifier2_0': model.classifier2_0.state_dict(),
                    'classifier2_1': model.classifier2_1.state_dict(),
                    'classifier2_2': model.classifier2_2.state_dict(),
                    'classifier2_3': model.classifier2_3.state_dict(),
                    'classifier2_4': model.classifier2_4.state_dict(),
                    'classifier2_5': model.classifier2_5.state_dict(),
                    'classifier2_6': model.classifier2_6.state_dict(),
                    'classifier2_7': model.classifier2_7.state_dict(),
                    'classifier3_0': model.classifier3_0.state_dict(),
                    'classifier3_1': model.classifier3_1.state_dict(),
                    'classifier3_2': model.classifier3_2.state_dict(),
                    'classifier3_3': model.classifier3_3.state_dict(),
                    'classifier3_4': model.classifier3_4.state_dict(),
                    'classifier3_5': model.classifier3_5.state_dict(),
                    'classifier3_6': model.classifier3_6.state_dict(),
                    'classifier3_7': model.classifier3_7.state_dict(),
                    'classifier4_0': model.classifier4_0.state_dict(),
                    'classifier4_1': model.classifier4_1.state_dict(),
                    'classifier4_2': model.classifier4_2.state_dict(),
                    'classifier4_3': model.classifier4_3.state_dict(),
                    'classifier4_4': model.classifier4_4.state_dict(),
                    'classifier4_5': model.classifier4_5.state_dict(),
                    'classifier4_6': model.classifier4_6.state_dict(),
                    'classifier4_7': model.classifier4_7.state_dict(),
                }, os.path.join(args.output_dir, 'pytorch_model' + str(epoch) + '.bin'))

                best_f1 = avg_f1
                early_stopping_count = 0

# 列名
                header = ["P", "R", "F1"]

                # 创建CSV文件并写入列名
                with open(args.data_dir + '_example.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

                # 打开CSV文件并单独赋值
                with open(args.data_dir + '_example.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([eval_precision_issue, eval_recall_issue, eval_f1_issue])
                    writer.writerow([eval_precision_pr, eval_recall_pr, eval_f1_pr])
                    writer.writerow([eval_precision_commit, eval_recall_commit, eval_f1_commit])
                    writer.writerow([eval_precision_code_comments, eval_recall_code_comments, eval_f1_code_comments])


                

            if avg_f1 < best_f1:
                early_stopping_count = early_stopping_count + 1

            if early_stopping_count >= args.patience:
                print("early_stopping..........................")
                break 


            result = {'eval_loss_issue': eval_loss_issue,
                      'eval_precision_issue': eval_precision_issue,
                      'eval_recall_issue': eval_recall_issue,
                      'eval_f1_issue': eval_f1_issue,
                      'eval_loss_request': eval_loss_pr,
                      'eval_precision_request': eval_precision_pr,
                      'eval_recall_request': eval_recall_pr,
                      'eval_f1_request': eval_f1_pr,
                      'eval_loss_commit': eval_loss_commit,
                      'eval_precision_commit': eval_precision_commit,
                      'eval_recall_commit': eval_recall_commit,
                      'eval_f1_commit': eval_f1_commit,
                      'eval_loss_code_comments': eval_loss_code_comments,
                      'eval_precision_code_comments': eval_precision_code_comments,
                      'eval_recall_code_comments': eval_recall_code_comments,
                      'eval_f1_code_comments': eval_f1_code_comments,
                      'avg_precision': avg_precision,
                      'avg_recall': avg_recall,
                      'avg_f1': avg_f1,
                      'global_step': global_step,
                      'loss': tr_loss / nb_tr_steps}

            output_eval_file = os.path.join(args.output_dir, "eval_results_ep_" + str(epoch) + ".txt")
            print("output_eval_file=", output_eval_file)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
