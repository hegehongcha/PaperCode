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
import torch.nn as nn


from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from utils import (weight_init, calc_iwnll, calc_rec, calc_mi, calc_au, BucketingDataLoader, TextDataset_Split, TextDataset_2Tokenizers, frange_cycle_linear, frange_cycle_zero_linear)


from modules import VAE


# logging.getLogger("azure").setLevel(logging.WARNING)
# logging.getLogger("TableService").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}


class Args(object):
    def __init__(self,
                 latent_size=32,
                 fb_mode=1,
                 dim_target_kl=0.5,
                 length_weighted_loss=1,
                 beta=1,
                 mh_burn_in=1,
                 mh_thin=1,
                 device="cuda"
                 ):
        self.latent_size = latent_size
        self.fb_mode = fb_mode
        self.dim_target_kl = dim_target_kl
        self.length_weighted_loss = length_weighted_loss
        self.beta = beta
        self.mh_burn_in = mh_burn_in
        self.mh_thin = mh_thin
        self.device = device


class MI_Network(nn.Module):
    """VAE with normal prior"""
    def __init__(self, decoder,args): #
        super(VAE, self).__init__()

        self.decoder = decoder
        self.args = args
        self.nz = args.latent_size
        self.linear = nn.Linear(args.nz, 2 * args.nz, bias=False)

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, labels, z):
        conditional_prob_mu, conditional_prob_logvar = self.linear(z).chunk(2,-1)
        latent_z = self.reparameterize(conditional_prob_mu, conditional_prob_logvar)
        outputs = self.decoder(input_ids=labels, past=latent_z, labels=labels)
        loss_rec = outputs[0]
        return loss_rec

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)


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
                text_sam = f.readlines()
                text_samples = [text_sam[j] for j in train_data_index]

            train_text_data = []
            titles = []
            for text_sample in text_samples:
                # train_sent = []
                # # train_sent.append(title)
                # sents = text_sample["sent"]
                # for sent in sents:
                #     train_sent.append(sent)
                # # train_sent = np.array(train_sent)
                train_text_data.append(text_sample)
        if text_path != None:
            yield train_text_data, rows_index
        else:
            yield None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="dd", help="[dd, cornellmovie, personachat, CMU_Dog]")

    config = parser.parse_args()
    encoder = BertForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased", latent_size=32)

    decoder = GPT2ForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/pytorch_model")
    # language_model = GPT2ForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/pytorch_model")


    # condition_encoder = BertForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased", latent_size=32)
    tokenizer_encoder = BertTokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/bert_tokenizer")
    tokenizer_decoder = GPT2Tokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/gpt2_tokenizer")
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    decoder.resize_token_embeddings(len(tokenizer_decoder))
    print(len(tokenizer_decoder))
    print(len(tokenizer_encoder))
    args = Args()
    model = VAE(encoder, decoder, tokenizer_encoder, tokenizer_decoder, args).cuda()
    min_index = 0
    max_index = 50000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    epoch = 0
    first = 1
    rows_index = 0
    rows = np.arange(min_index, max_index)
    rows_dict = {}
    np.random.shuffle(rows)

    eos_token_id = tokenizer_decoder.encode("<EOS>")[0]
    cls_token_id = tokenizer_encoder.encode("[CLS]")[0]
    sep_token_id = tokenizer_encoder.encode("[SEP]")[0]
    bos_token_id = tokenizer_decoder.encode("<BOS>")[0]
    while epoch <= 40:
        torch.cuda.empty_cache()
        try:
            checkpoint = torch.load(f"./{config.datasets}/NLG_evaluation.pkl")
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
            checkpoint = torch.load(f"./{config.datasets}/NLG_evaluation.pkl")
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
            with open(f"./{config.datasets}/NLG_evaluation.json", "w", encoding="utf-8") as f:
                f.write(rows_json)
            print(555)
        else:
            rows_index = checkpoint["rows_index"]
            i = checkpoint['iter']
            #             epoch = checkpoint["epoch"]
            print(666)

            with open(f"./{config.datasets}/NLG_evaluation.json", "r", encoding="utf-8") as f:
                rows_dict_str = f.read()
                rows_dict = json.loads(rows_dict_str)
            rows = rows_dict['rows']
        rows = np.array(rows)
        train_data_gen = Dataloader(text_path=f"./{config.datasets}/datasets.txt",
                                    batch_size=64,
                                    min_index=min_index,
                                    max_index=max_index,
                                    rows_index=rows_index,
                                    rows=rows)

        for train_data in train_data_gen:
            rec_loss_list = []
            loss_kl_list = []
            loss_list = []

            texts = np.array(train_data[0])
            # print(texts)
            rows_index = train_data[1]
            # condition_inputs_list = tokenizer_encoder.encode(texts)
            for text in texts:
                label_text = text.split("[SEP]")[1].strip()
                condition_text = text.split("[SEP]")[0].strip()
                condition_inputs_list = tokenizer_encoder.encode(condition_text)
                label_inputs_list = tokenizer_decoder.encode(label_text)
                raw_inputs_list = tokenizer_encoder.encode(text)
                inputs_list = tokenizer_encoder.encode(label_text)
                inputs_list.insert(0, cls_token_id)
                label_inputs_list.insert(0, bos_token_id)
                label_inputs_list.append(eos_token_id)
                # print(len(label_inputs_list))
                if len(raw_inputs_list) <= 300:
                    inputs = torch.from_numpy(np.array(inputs_list)).long().cuda().unsqueeze(0)
                    labels = torch.from_numpy(np.array(label_inputs_list)).long().cuda().unsqueeze(0)
                    conditions = torch.from_numpy(np.array(condition_inputs_list)).long().cuda().unsqueeze(0)

                    rec_loss, loss_kl, loss = model(inputs, labels, conditions)
                    # print(rec_loss, loss_kl, loss)
                    rec_loss_list.append(rec_loss)
                    loss_kl_list.append(loss_kl)
                    loss_list.append(loss)

            rec_loss = torch.mean(torch.stack(rec_loss_list))
            loss_kl = torch.mean(torch.stack(loss_kl_list))
            loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:", epoch, "iter:", i, "loss:", loss.item(), "loss_kl:", loss_kl.item(), "rec_loss:",
                  rec_loss.item(), "rows:", rows_index, f"./{config.datasets}/NLG_evaluation")
            i = i + 1
            if i % 10 == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "rows_index": rows_index,
                              "flag": 0,
                              "epoch": epoch,
                              "iter": i,
                              "first": first}
                torch.save(checkpoint, f"./{config.datasets}/NLG_evaluation.pkl")

        print(777)
        epoch = epoch + 1
        checkpoint = {"model_state_dict": model.state_dict(),
                      "rows_index": rows_index,
                      "flag": 1,
                      "epoch": epoch,
                      "iter": 0,
                      "first": 0}
        torch.save(checkpoint, f"./{config.datasets}/NLG_evaluation.pkl")

main()