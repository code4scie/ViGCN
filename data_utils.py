# -*- coding: utf-8 -*-


import os
import sys
sys.path.append(r'./LAL-Parser/src_joint')
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
import pickle
from torch.utils.data import Dataset,DataLoader
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from transformers import BertModel,AdamW,BertTokenizer

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                # position
                aspect_post = [aspect['from'], aspect['to']] 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data

class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w  
    
    def word_to_id(self, word):  
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):   
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length
    
    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

class ABSAGCNDATA(Dataset):
  def __init__(self,fname,tokenizer,opt):
    self.data = []
    parse = ParseData
    polarity_dict = {'positive':0,'negative':1,'neutral':2}
    for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
      polarity = polarity_dict[obj['label']]
      text = obj['text']
      term = obj['aspect']
      term_start = obj['aspect_post'][0]
      term_end = obj['aspect_post'][1]
      text_list = obj['text_list']
      left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]

      #polarity score
      senticNet = {}
      with open('./senticnet/senticnetword.txt','r') as file:
        fp = file
        for line in fp:
          line = line.strip()
          if not line:
            continue
          word,sentic = line.split('\t')
          senticNet[word] = sentic
        fp.close()
      senticvalue = np.zeros(context_len)
      for word_i,bword in enumerate(text_list):
        if str(bword) in senticNet:
          sentic = 1+abs(float(senticNet[str(bword)]))
        else:
          sentic = 1
        senticvalue[word_i] = sentic
      senticvalue = torch.from_numpy(senticvalue)

      #SRD
      asp_avg_index = (len(left)*2+len(term)-1)/2
      context_len = len(text_list)
      for i in range(context_len):
        srd = abs(i-asp_avg_index)-len(term)/2
        if srd > opt.SRD:
          senticvalue[i] = senticvalue[i]
        else:
          senticvalue[i] = senticvalue[i]+1
      
      
      left_tokens, term_tokens, right_tokens = [], [], []
      left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
      left_sentic, term_sentic, right_sentic = [], [], [] 
      for ori_i, w in enumerate(left):
          for t in tokenizer.tokenize(w):
              left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
              left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
              left_sentic.append(senticvalue[ori_i])
      asp_start = len(left_tokens)  
      offset = len(left) 
      for ori_i, w in enumerate(term):        
          for t in tokenizer.tokenize(w):
              term_tokens.append(t)
              term_tok2ori_map.append(ori_i + offset)
              term_sentic.append(senticvalue[ori_i+ offset])
      asp_end = asp_start + len(term_tokens)
      offset += len(term) 
      for ori_i, w in enumerate(right):
          for t in tokenizer.tokenize(w):
              right_tokens.append(t)
              right_tok2ori_map.append(ori_i+offset)
              term_sentic.append(senticvalue[ori_i+ offset])
      while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
          if len(left_tokens) > len(right_tokens):
              left_tokens.pop(0)
              left_tok2ori_map.pop(0)
          else:
              right_tokens.pop()
              right_tok2ori_map.pop()
              
      bert_tokens = left_tokens + term_tokens + right_tokens
      tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
      sentic_token = left_sentic + term_sentic + right_sentic
      truncate_tok_len = len(bert_tokens)
      tok_adj = np.zeros(
        (truncate_tok_len, truncate_tok_len), dtype='float32')
      for i in range(truncate_tok_len):
          for j in range(truncate_tok_len):
              tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]

         
      # context raw 
      context_raw_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(bert_tokens)+[tokenizer.sep_token_id]
      context_raw_asp_len = len(context_raw_asp_ids)
      paddings_raw = [0] * (tokenizer.max_seq_len - context_raw_asp_len)
      context_raw_asp_ids += paddings_raw
      context_raw_asp_ids = np.asarray(context_raw_asp_ids,dtype='int64')
      #context+aspect
      context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
          bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
      context_asp_len = len(context_asp_ids)
      paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
      context_len = len(bert_tokens)
      context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
      #src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
      src_mask = [0] + [1]*context_len + [0] * (85 - context_len - 1)
      aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
      aspect_mask = aspect_mask + (85 - len(aspect_mask)) * [0]
      context_asp_attention_mask = [1] * context_asp_len + paddings
      context_asp_ids += paddings
      #aspect bert 
      aspect_bert_indices = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(term_tokens) + [tokenizer.sep_token_id]
      aspect_bert_indices_len = len(aspect_bert_indices)
      paddings_abi = [0]*(tokenizer.max_seq_len-aspect_bert_indices_len)
      aspect_bert_indices += paddings_abi
      aspect_bert_indices = np.asarray(aspect_bert_indices,dtype='int64')
      #to numpy
      context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
      context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
      context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
      src_mask = np.asarray(src_mask, dtype='int64')
      aspect_mask = np.asarray(aspect_mask,dtype='int64')

      # pad adj
      context_asp_adj_matrix = np.zeros(
          (tokenizer.max_seq_len+1, tokenizer.max_seq_len+1)).astype('float32')
      context_asp_adj_matrix[1:context_len + 1, 1:context_len + 1] = tok_adj
      context_asp_adj_matrix[1:context_len+1,tokenizer.max_seq_len] = sentic_token
      context_asp_adj_matrix[tokenizer.max_seq_len,1:context_len+1] = sentic_token
      data = {
          'text_bert_indices': context_asp_ids,
          'text_raw_bert_indices':context_raw_asp_ids,
          'aspect_bert_indices': aspect_bert_indices,
          'bert_segments_ids': context_asp_seg_ids,
          'attention_mask': context_asp_attention_mask,
          'asp_start': asp_start,
          'asp_end': asp_end,
          'adj_matrix': context_asp_adj_matrix,
          'src_mask': src_mask,
          'aspect_mask': aspect_mask,
          'polarity': polarity,
      }
      self.data.append(data)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]
