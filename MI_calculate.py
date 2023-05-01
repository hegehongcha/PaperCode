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

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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
    def __init__(self, decoder, args): #
        super(MI_Network, self).__init__()

        self.decoder = decoder
        self.args = args
        self.nz = args.latent_size
        self.linear = nn.Linear(self.nz, 2 * self.nz, bias=False)

        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, labels, z, condition_pooled_hidden_fea):
        conditional_prob_mu, conditional_prob_logvar = self.linear(z).chunk(2,-1)
        latent_z = self.reparameterize(conditional_prob_mu, conditional_prob_logvar)
        # print(labels.shape, latent_z.shape)
        latent_z = latent_z.squeeze(0)
        latent_z = torch.cat([condition_pooled_hidden_fea, latent_z], 1)
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
    condition_encoder = BertForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/bert-base-uncased", latent_size=32)
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
    language_model= GPT2ForLatentConnector.from_pretrained("/cuixiaohui/zk/Optimus-master/pytorch_model")
    tokenizer_language_model = GPT2Tokenizer.from_pretrained("/cuixiaohui/zk/Optimus-master/gpt2_tokenizer")
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_language_model.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens to GPT2')
    language_model.resize_token_embeddings(len(tokenizer_language_model))
    print(len(tokenizer_language_model))
    min_index = 0
    max_index = 40000
    MI_model = MI_Network(language_model, args).cuda()

    checkpoint = torch.load(f"./{config.datasets}/NLG_evaluation.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])
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
    i=0
    learning_rate = 0.001
    while epoch <= 300:
        torch.cuda.empty_cache()
        try:
            save_model = torch.load(f"./{config.datasets}/MI_model.pkl")
            # print(checkpoint_1["model_state_dict"])
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
            save_model = torch.load(f"./{config.datasets}/MI_model.pkl")
            # save_model = t.load(path)
            model_dict = MI_model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(state_dict)
            MI_model.load_state_dict(model_dict)
            # MI_model.load_state_dict(save_model["model_state_dict"])
            flag = checkpoint["flag"]
            #             first = 0
            #             rows_index = checkpoint["rows_index"]
            #             i = checkpoint['iter']
            epoch = save_model["epoch"]
            print(444)
        if flag:
            i = 0
            rows_index = 0
            rows = np.arange(min_index, max_index)
            rows_dict = {}
            np.random.shuffle(rows)
            rows_dict["rows"] = rows.tolist()
            rows_json = json.dumps(rows_dict)
            with open(f"./{config.datasets}/MI_model.json", "w", encoding="utf-8") as f:
                f.write(rows_json)
            print(555)
        else:
            rows_index = checkpoint_1["rows_index"]
            i = save_model['iter']
            #             epoch = checkpoint["epoch"]
            print(666)

            with open(f"./{config.datasets}/MI_model.json", "r", encoding="utf-8") as f:
                rows_dict_str = f.read()
                rows_dict = json.loads(rows_dict_str)
            rows = rows_dict['rows']
        rows = np.array(rows)
        train_data_gen = Dataloader(text_path=f"./{config.datasets}/datasets.txt",
                                    batch_size=32,
                                    min_index=min_index,
                                    max_index=max_index,
                                    rows_index=rows_index,
                                    rows=rows)


        # i = 0
        if epoch % 50 == 0:
            learning_rate = learning_rate/10
        optimizer = torch.optim.Adam(MI_model.parameters(), lr=learning_rate)
        for train_data in train_data_gen:
            texts = np.array(train_data[0])
            # print(texts)
            rows_index = train_data[1]
            # condition_inputs_list = tokenizer_encoder.encode(texts)
            loss_rec_list = []
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
                if len(raw_inputs_list) <= 300:
                    # print(len(raw_inputs_list))
                    inputs = torch.from_numpy(np.array(inputs_list)).long().cuda().unsqueeze(0)
                    labels = torch.from_numpy(np.array(label_inputs_list)).long().cuda().unsqueeze(0)
                    conditions = torch.from_numpy(np.array(condition_inputs_list)).long().cuda().unsqueeze(0)
                    with torch.no_grad():
                        raw_attention_mask = (inputs > 0).float()
                        condition_attention_mask = (conditions > 0).float()

                        reconstrution_mask = (labels != -1).float()  # 50257 is the padding token for GPT2
                        sent_length = torch.sum(reconstrution_mask, dim=1)

                        raw_outputs = model.encoder(inputs, raw_attention_mask)
                        raw_pooled_hidden_fea = raw_outputs[1]  # model outputs are always tuple in pytorch-transformers (see doc)
                        condition_outputs = model.encoder(conditions, condition_attention_mask)
                        condition_pooled_hidden_fea = condition_outputs[1]
                        c_x = torch.cat([condition_pooled_hidden_fea, raw_pooled_hidden_fea], 1)
                        # print(c_x.shape)
                        # prior_mu, prior_logvar = self.encoder.prior_linear(condition_pooled_hidden_fea).chunk(2, -1)
                        posterior_mu, posterior_logvar = model.encoder.linear(c_x).chunk(2, -1)
                        latent_z = model.reparameterize(posterior_mu, posterior_logvar, nsamples=1)
                        latent_z = latent_z.squeeze(1)
                        # latent_z = torch.cat([condition_pooled_hidden_fea, latent_z], 1)
                    # print(latent_z.shape, labels.shape)
                    loss_rec = MI_model(labels, latent_z, condition_pooled_hidden_fea)
                    loss_rec_list.append(loss_rec)
            loss_rec = torch.mean(torch.stack(loss_rec_list))
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()
            print("epoch:", epoch, "iter:", i, "rec_loss:", loss_rec.item(), "rows:", rows_index, f"./{config.datasets}/MI_calculate")
            i = i + 1
            if i % 10 == 0:
                checkpoint = {"model_state_dict": MI_model.state_dict(),
                              "rows_index": rows_index,
                              "flag": 0,
                              "epoch": epoch,
                              "iter": i,
                              "first": first}
                torch.save(checkpoint, f"./{config.datasets}/MI_model.pkl")
        print(777)
        epoch = epoch + 1
        checkpoint = {"model_state_dict": MI_model.state_dict(),
                      "rows_index": rows_index,
                      "flag": 1,
                      "epoch": epoch,
                      "iter": 0,
                      "first": 0}
        torch.save(checkpoint, f"./{config.datasets}/MI_model.pkl")

main()