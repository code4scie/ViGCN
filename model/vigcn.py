

import json
import argparse
from collections import Counter
import re
import pickle
from tqdm import tqdm
from transformers import BertTokenizer
import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from transformers import BertModel,AdamW

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class globalGCNBert(nn.Module):
  def __init__(self,bert,opt,num_layers):
    super(globalGCNBert,self).__init__()
    self.bert = bert
    self.opt = opt
    self.layers = num_layers
    self.mem_dim = opt.bert_dim  #// 2
    self.bert_dim = opt.bert_dim
    self.bert_drop = nn.Dropout(opt.bert_dropout) #0.3
    self.pooled_drop = nn.Dropout(opt.bert_dropout) #0.3
    self.gcn_drop = nn.Dropout(opt.gcn_dropout) #0.1
    self.layernorm = LayerNorm(opt.bert_dim)

    self.W = nn.ModuleList()
    for layer in range(self.layers):
      input_dim = self.bert_dim if layer == 0 else self.mem_dim
      self.W.append(nn.Linear(input_dim, self.mem_dim))

  def forward(self,adj,inputs):
    text_bert_indices,text_raw_bert_indices,aspect_bert_indices,bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
    src_mask = src_mask.unsqueeze(-2)
      
    sequence_output = self.bert(text_bert_indices, attention_mask=attention_mask,token_type_ids=bert_segments_ids)[0]
    super_node = torch.mean(sequence_output,dim=1).unsqueeze(1)
    super_node = torch.zeros_like(super_node)
    pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask,token_type_ids=bert_segments_ids)[1]
    gcn_inputs = self.layernorm(sequence_output)
    gcn_inputs = self.bert_drop(gcn_inputs)
    pooled_output = self.pooled_drop(pooled_output)
    gcn_inputs = torch.cat((gcn_inputs,super_node),1)
    
    texts = text_raw_bert_indices.cpu().numpy()
    asps = aspect_bert_indices.cpu().numpy()
         
    denom_dep = adj.sum(2).unsqueeze(2) + 1  
    outputs_dep = gcn_inputs

    for l in range(self.layers):
      Ax_dep = adj.bmm(outputs_dep)
      AxW_dep = self.W[l](Ax_dep)
      AxW_dep = AxW_dep / denom_dep
      gAxW_dep = F.relu(AxW_dep)
      outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
    return outputs_dep,pooled_output

class globalGCNAbsaModel(nn.Module):
  def __init__(self,bert,opt):
    super().__init__()
    self.opt = opt
    self.gcn = globalGCNBert(bert,opt,opt.num_layers)
  def forward(self,inputs):
    text_bert_indices,text_raw_bert_indices,aspect_bert_indices,bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
    h,pooled_output = self.gcn(adj_dep,inputs)
    h = h[:,:opt.max_length,:].to('cuda')

    # avg pooling asp feature
    asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
    aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
    outputs = (h*aspect_mask).sum(dim=1)/asp_wn
    #return outputs,adj_dep
    return outputs,adj_dep,pooled_output

class GCNBertClassifier(nn.Module):
  def __init__(self,bert,opt):
    super().__init__()
    self.opt = opt
    self.gcn_model = globalGCNAbsaModel(bert,opt=opt)
    self.classifier = nn.Linear(opt.hidden_dim,opt.polarities_dim)
  def forward(self,inputs):
    outputs,adj_dep,pooled_output = self.gcn_model(inputs)
    logits = self.classifier(outputs)
    return logits,None
