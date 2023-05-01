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
                text_sam = f.read().split('\n')
                text_samples = [text_sam[j] for j in train_data_index]

            train_text_data = text_samples
        if text_path != None:
            yield train_text_data, rows_index
        else:
            yield None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="dd", help="[dd, cornellmovie, personachat, CMU_Dog]")

    config = parser.parse_args()
    model = BertForNextSentencePrediction.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased").cuda()
    tokenizer = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
    min_index = 0
    max_index = 50000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epoch = 0
    first = 1
    rows_index = 0
    rows = np.arange(min_index, max_index)
    rows_dict = {}
    np.random.shuffle(rows)
    cls_token_id = tokenizer.encode("[CLS]")[0]
    sep_token_id = tokenizer.encode("[SEP]")[0]
    with open(f"./{config.datasets}/datasets.txt", "r", encoding="utf-8") as f:
        text_sam = f.read().split('\n')

    while epoch <= 100:
        torch.cuda.empty_cache()
        try:
            checkpoint = torch.load(f"./{config.datasets}/NSP.pkl")
            first = 0
            print("111")
        except:
            first = 1
            print("222")
        if first == 1:
            flag = 1
            first = 0
            i = 0
            epoch = 0
            print(333)
        else:
            checkpoint = torch.load(f"./{config.datasets}/NSP.pkl")
            model.load_state_dict(checkpoint["model_state_dict"])
            flag = checkpoint["flag"]
            #             first = 0
            #             rows_index = checkpoint["rows_index"]
            #             i = checkpoint['iter']
            epoch = checkpoint["epoch"]
            print(444)
        if flag:
            i = 0
            rows_index = 0
            rows = np.arange(min_index, max_index)
            rows_dict = {}
            np.random.shuffle(rows)
            rows_dict["rows"] = rows.tolist()
            rows_json = json.dumps(rows_dict)
            with open(f"./{config.datasets}/NSP.json", "w", encoding="utf-8") as f:
                f.write(rows_json)
            print(555)
        else:
            rows_index = checkpoint["rows_index"]
            i = checkpoint['iter']
            #             epoch = checkpoint["epoch"]
            print(666)

            with open(f"./{config.datasets}/NSP.json", "r", encoding="utf-8") as f:
                rows_dict_str = f.read()
                rows_dict = json.loads(rows_dict_str)
            rows = rows_dict['rows']
        rows = np.array(rows)
        train_data_gen = Dataloader(text_path=f"./{config.datasets}/datasets.txt",
                                    batch_size=128,
                                    min_index=min_index,
                                    max_index=max_index,
                                    rows_index=rows_index,
                                    rows=rows)

        for batch in train_data_gen:
            #     print(batch[0])
            loss_list = []
            rows_index = batch[1]
            for train_data in batch[0]:
                sentence_a = train_data.split("[SEP]")[0].strip()
                prob = random.uniform(0, 1)
                if prob > 0.5:
                    sentence_b = train_data.split("[SEP]")[1].strip()
                    label = 1
                else:
                    while True:
                        try:
                            index = random.randint(0, len(text_sam)-1)
                            if text_sam[index] != train_data:
                                sentence_b = text_sam[index].split("[SEP]")[1].strip()
                                label = 0
                                break
                        except:
                            pass
                # label = int(train_data["label"])
                # print(sentence_a, sentence_b, label)
                tokenize_a = tokenizer.encode(sentence_a)
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
                    label = torch.LongTensor([label]).cuda()
                    segments_tensor = torch.from_numpy(np.array(segments_tensor)).long().cuda().unsqueeze(0)
                    mask_tensor = torch.from_numpy(np.array(mask_tensor)).long().cuda().unsqueeze(0)
                    outputs = model(inputs, attention_mask=mask_tensor, token_type_ids=segments_tensor,
                                    next_sentence_label=label)
                    loss = outputs[0]
                    loss_list.append(loss)

            loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:", epoch, "iter:", i, "loss:", loss.item(), "rows:", rows_index, f"./{config.datasets}/NSP")
            i = i + 1
            if i % 10 == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "rows_index": rows_index,
                              "flag": 0,
                              "epoch": epoch,
                              "iter": i,
                              "first": first}
                torch.save(checkpoint, f"./{config.datasets}/NSP.pkl")

        print(777)
        epoch = epoch + 1
        checkpoint = {"model_state_dict": model.state_dict(),
                      "rows_index": rows_index,
                      "flag": 1,
                      "epoch": epoch,
                      "iter": 0,
                      "first": 0}
        torch.save(checkpoint, f"./{config.datasets}/NSP.pkl")


main()



