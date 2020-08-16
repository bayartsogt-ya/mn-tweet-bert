# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange

import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from modeling import BertForSequenceClassification
from tokenization import BertTokenizer
from optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "bert_config.json"

logger = logging.getLogger(__name__)

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def load_model():
    task_name = "tweetmn"
    # data_dir = 
    bert_model = "./models/uncased_bert_base_pytorch"
    output_dir = "./output/tweetmn-epoch-10/"

    # Other parameters
    cache_dir = ""
    max_seq_length = 128
    do_train = True
    do_lower_case = True
    eval_batch_size = 8
    learning_rate = 5e-5
    num_train_epochs = 3.0
    warmup_proportion = 0.1
    no_cuda = True
    overwrite_output_dir = True
    local_rank = -1
    seed = 42
    gradient_accumulation_steps = 1
    fp16 = True
    loss_scale = 0
    server_ip = ''
    server_port = ''

    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            argsgradient_accumulation_steps))


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if not os.path.exists(output_dir) and local_rank in [-1, 0]:
        os.makedirs(output_dir)

    task_name = task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    # model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)

    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    if local_rank == 0:
        torch.distributed.barrier()

    # if fp16:
    #     model.half()
    model.to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    # Load a trained model and vocabulary that you have fine-tuned
    
    # model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
    model.to(device)

    return model, tokenizer, processor, output_mode, device

def predict(input_text, model, tokenizer, processor, output_mode, device):

    # Parameter
    max_seq_length = 128
    eval_batch_size = 8

    ### Evaluation
    label_list = processor.get_labels()
    print("input text: ", input_text)
    eval_examples = processor.get_test_example(input_text)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, output_mode)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    preds = []
    out_label_ids = None

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    preds = preds[0]

    print(preds.shape)
    print("preds", preds)

    scores = softmax(preds[0])
    print("scores", scores)

    arg_max = np.argmax(preds)
    print("Argmax",arg_max)

    return arg_max, scores[arg_max] # returning the index of labels and its score

if __name__ == "__main__":
    main()
