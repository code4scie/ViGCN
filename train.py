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
from model.vigcn import GCNBertClassifier
from data_utils import Tokenizer4BertGCN,ABSAGCNDATA

"""model

train
"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True

class Instructor:
  def __init__(self,opt):
    self.opt = opt
    tokenizer = Tokenizer4BertGCN(85, 'bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    #model_class = LocalGCNBertClassifier(bert,2)
    self.model = GCNBertClassifier(bert,opt).to('cuda')
    trainset = ABSAGCNDATA(opt.dataset_files['train'], tokenizer, opt=opt)
    testset = ABSAGCNDATA(opt.dataset_files['test'], tokenizer, opt=opt)
    self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
    self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)
    
    if opt.device.type == 'cuda':
      logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
    self._print_args()

  def _print_args(self):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in self.model.parameters():
      n_params = torch.prod(torch.tensor(p.shape))
      if p.requires_grad:
        n_trainable_params += n_params
      else:
        n_nontrainable_params += n_params

    logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
    logger.info('training arguments:')
    
    for arg in vars(self.opt):
      logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
  def _reset_params(self):
    for p in self.model.parameters():
      if p.requires_grad:
        if len(p.shape) > 1:
          #self.opt.initializer(p)   # xavier_uniform_
          torch.nn.init.xavier_uniform_(p)
        else:
          stdv = 1. / (p.shape[0]**0.5)
          torch.nn.init.uniform_(p, a=-stdv, b=stdv)
  def get_bert_optimizer(self,model):
    no_decay = ['bias','LayerNorm.weight']
    diff_part = ["bert.embeddings","bert.encoder"]
    logger.info("bert learning rate on")
    #quan zhong shuai jian: weight_decay,guo ni he 
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': self.opt.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)
    return optimizer

  def _train(self,criterion,optimizer,max_test_acc_overall=0):
    max_test_acc = 0
    max_f1 = 0
    global_step = 0
    model_path = ''
    for epoch in range(self.opt.num_epoch):
      logger.info('>' * 60)
      logger.info('epoch: {}'.format(epoch))
      n_correct, n_total = 0, 0
      for i_batch, sample_batched in enumerate(self.train_dataloader):
        global_step += 1
        # switch model to training mode, clear gradient accumulators
        self.model.train()
        optimizer.zero_grad()
        inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
        outputs, penal = self.model(inputs)
        targets = sample_batched['polarity'].to(self.opt.device)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        
        if global_step % self.opt.log_step == 0:
          n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
          n_total += len(outputs)
          train_acc = n_correct / n_total
          test_acc, f1 = self._evaluate()
          if test_acc > max_test_acc:
            max_test_acc = test_acc
            if test_acc > max_test_acc_overall:
              if not os.path.exists('./state_dict'):
                os.mkdir('./state_dict')
              model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format("vigcn", "restaurants", test_acc, f1)
              self.best_model = copy.deepcopy(self.model)
              logger.info('>> saved: {}'.format(model_path))
          if f1 > max_f1:
            max_f1 = f1
          logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
    return max_test_acc, max_f1, model_path
  def _evaluate(self, show_results=False):
    #switch model to evaluation mode
    self.model.eval()
    n_test_correct, n_test_total = 0, 0
    targets_all, outputs_all = None, None
    with torch.no_grad():
      for batch, sample_batched in enumerate(self.test_dataloader):
        inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
        targets = sample_batched['polarity'].to(self.opt.device)
        outputs, penal = self.model(inputs)
        n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
        n_test_total += len(outputs)
        targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
        outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
    test_acc = n_test_correct / n_test_total
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

    labels = targets_all.data.cpu()
    predic = torch.argmax(outputs_all, -1).cpu()
    if show_results:
      report = metrics.classification_report(labels, predic, digits=4)
      confusion = metrics.confusion_matrix(labels, predic)
      return report, confusion, test_acc, f1

    return test_acc, f1
  
  def _test(self):
    self.model = self.best_model
    self.model.eval()
    test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
    logger.info("Precision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...")
    logger.info(test_confusion)
  
  def run(self):
    criterion = nn.CrossEntropyLoss()
    optimizer = self.get_bert_optimizer(self.model)
    max_test_acc_overall = 0
    max_f1_overall = 0
    max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
    logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
    max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
    max_f1_overall = max(max_f1, max_f1_overall)
    torch.save(self.best_model.state_dict(), model_path)
    logger.info('>> saved: {}'.format(model_path))
    logger.info('#' * 60)
    logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
    logger.info('max_f1_overall:{}'.format(max_f1_overall))
    self._test()

dataset_files = {
     'restaurant': {
              'train': './dataset/Restaurants_corenlp/train.json',
              'test': './dataset/Restaurants_corenlp/test.json',
          },
    'laptop': {
              'train': './dataset/Laptops_corenlp/train.json',
              'test': './dataset/Laptops_corenlp/test.json'
          },
    'twitter': {
              'train': './dataset/Tweets_corenlp/train.json',
              'test': './dataset/Tweets_corenlp/test.json',
          }
    }

inputs_colses = ['text_bert_indices','text_raw_bert_indices','aspect_bert_indices','bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix', 'src_mask', 'aspect_mask']

initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
}
optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
}

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--model_name', default='bert', type=str)
parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--l2reg', default=1e-4, type=float)
parser.add_argument('--num_epoch', default=15, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--log_step', default=5, type=int)
parser.add_argument('--embed_dim', default=300, type=int)
parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=768, help='GCN mem dim.') #384
parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
parser.add_argument('--polarities_dim', default=3, type=int, help='3')

parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
parser.add_argument('--lower', default=True, help='Lowercase all words.')
parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
parser.add_argument('--loop', default=True)

parser.add_argument('--max_length', default=100, type=int)
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
parser.add_argument('--seed', default=1000, type=int) #0
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument('--pad_id', default=0, type=int)
parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--losstype', default=None, type=str, help="['doubleloss', 'orthogonalloss', 'differentiatedloss']")
parser.add_argument('--alpha', default=0.3, type=float) #0.4 0.9
parser.add_argument('--beta', default=0.3, type=float) #0.3 0.6 

# * bert
parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--bert_dim', type=int, default=768)
parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
parser.add_argument('--diff_lr', default=False, action='store_true')
parser.add_argument('--bert_lr', default=2e-5, type=float)
opt = parser.parse_args(args=[])

opt.dataset_file = dataset_files[opt.dataset]
opt.inputs_cols = inputs_colses
opt.initializer = initializers[opt.initializer]
opt.optimizer = optimizers[opt.optimizer]
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
setup_seed(opt.seed)

if not os.path.exists('./log'):
        os.makedirs('./log', mode=0o777)
log_file = '{}-{}-{}.log'.format("vigcn",  "restaurants", strftime("%Y-%m-%d_%H:%M:%S", localtime()))
logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

ins = Instructor(opt)
ins.run()
