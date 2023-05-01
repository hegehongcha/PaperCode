from __future__ import absolute_import, division, print_function


import pdb
import argparse
import glob
import logging

import os
import pickle
import random

import numpy as np
import torch
import json
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from collections import Counter


from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule, BertForNextSentencePrediction,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)


from modules import VAE

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def Dataloader(text_path, batch_size, min_index, max_index, rows_index, rows):
    # rows = np.arange(min_index, max_index)
    # if shuffle:
    #     np.random.shuffle(rows)
    i = rows_index

    while i < max_index:
        # print(i)
        rows_index = i + batch_size
        if rows_index < max_index:
            train_data_index = [rows[x] for x in range(i-min_index, rows_index-min_index)]
        else:
            train_data_index = [rows[x] for x in range(i-min_index, max_index-min_index)]
        i = rows_index
        # print(train_data_index)
        if text_path != None:
            with open(text_path, "r", encoding="utf-8") as f:
                text_sam = json.load(f)
                text_samples = [text_sam[j] for j in train_data_index]

            train_text_data = text_samples
        if text_path != None:
            yield train_text_data, rows_index
        else:
            yield None

def mlog(s, config):
    if not os.path.exists(f"./{config.datasets}/log"):
        os.makedirs(f"./{config.datasets}/log")

    with open(f"./{config.datasets}/log/NSP.log", "a+", encoding="utf-8") as log_f:
        log_f.write(s+"\n")
    try:
        print(s)
    except:
        pass

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="dd", help="[dd, cornellmovie, personachat, CMU_Dog]")

    config = parser.parse_args()
    model = BertForNextSentencePrediction.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased").cuda()
    tokenizer = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
    checkpoint = torch.load(f"./{config.datasets}/NSP.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])
    cls_token_id = tokenizer.encode("[CLS]")[0]
    sep_token_id = tokenizer.encode("[SEP]")[0]

    with open(f"./{config.datasets}/s2s_eval_samples.json", "r", encoding="utf-8") as f:
        s2s_samples = json.load(f)

    with open(f"./{config.datasets}/hred_eval_samples.json", "r", encoding="utf-8") as f:
        hred_samples = json.load(f)

    with open(f"./{config.datasets}/datasets.txt", "r", encoding="utf-8") as fp:
        texts = fp.read().split('\n')

    # result = defaultdict(str)
    # with open(text_path, "r", encoding="utf-8") as f:
    #     text_sam = f.read().split('\n')
    #     text_samples = [text_sam[j] for j in train_data_index]
    num_sents = len(s2s_samples["samples"])
    num_accurate = 0
    for m, batch in enumerate(s2s_samples["samples"]):

        label_text = batch["reference"][1].strip()
        condition_text = batch["context"][0][1].strip()

        sentence_a = condition_text

        # s2s
        # sentence_b = batch["hypothesis"][1].strip()
        sentence_b = label_text
        # log_s = \
        #     f"conditions: {condition_text} \n" \
        #     f"s2s_generated_text: {sentence_b}\n"
        # mlog(log_s, config)
        # print(sentence_a, sentence_b, label)
        tokenize_a = tokenizer.encode(sentence_a)
        tokenize_a.insert(0, cls_token_id)
        tokenize_a.append(sep_token_id)
        len_a = len(tokenize_a)
        tokenize_b = tokenizer.encode(sentence_b)
        tokenize_b.append(sep_token_id)
        len_b = len(tokenize_b)
        inputs_ids = tokenize_a + tokenize_b
        # print(tokenize_a, tokenize_b)
        if len_a + len_b <= 512:
            segments_tensor = [0] * len_a + [1] * len_b
            mask_tensor = [1] * (len_a + len_b)
            inputs = torch.from_numpy(np.array(inputs_ids)).long().cuda().unsqueeze(0)
            # label = torch.LongTensor([label]).cuda()
            segments_tensor = torch.from_numpy(np.array(segments_tensor)).long().cuda().unsqueeze(0)
            mask_tensor = torch.from_numpy(np.array(mask_tensor)).long().cuda().unsqueeze(0)
            outputs = model(inputs, attention_mask=mask_tensor, token_type_ids=segments_tensor)
            predict = outputs[0]
            predict = predict.detach().cpu().numpy()
            pred = np.argmax(predict, axis=1).flatten()
            if int(pred[0]) == 1:
                num_accurate += 1
            # log_s = \
            #     f"prob: {pred[0]} \n"
            # mlog(log_s, config)
    print("accuracy rate: ", num_accurate/num_sents)
    print(num_sents, num_accurate)


main()



