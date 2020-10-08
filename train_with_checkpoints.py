#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nntools as nt


# In[2]:


from models.architectures import *
import torch
import torch.nn as nn
from config import args
import os
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


encoder = args['encoder'](args['embed_size']).to(device)
decoder = args['decoder'](args['embed_size'], args['hidden_size'], args['vocab_size'], args['num_layers'], list(args['vocabulary'].keys()), args['glove_path'], args.get('max_sentence_length', 100), args.get('is_pretrained', True)).to(device)


# In[5]:


criterion = args['loss_criterion']


# In[6]:


params = list(list(encoder.parameters()) + list(decoder.parameters()))


# In[7]:



optimizer = torch.optim.Adam(params, lr=args['learning_rate'])


# In[8]:


stats_manager = nt.StatsManager()


# In[9]:


exp1 = nt.Experiment(encoder, decoder, device, criterion, optimizer, stats_manager, 
                     output_dir=args['model_path'], perform_validation_during_training=True)


# In[10]:
import time


exp1.run(num_epochs=args['epochs'])
# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()

# a,b  = exp1.evaluate(mode='test',generate=False,generate_mode='deterministic',temperature=1)
# print((a,b))
#a,b  = exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=0.2)
# time.sleep(30)


# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()
# exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=0.1)
# time.sleep(30)

# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()
# exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=0.5)
# time.sleep(30)

# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()
# exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=1)
# time.sleep(30)

# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()
# exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=1.5)
# time.sleep(30)

# exp1.load_bestmodel()
# exp1.encoder.eval()
# exp1.decoder.eval()
# exp1.evaluate(mode='test',generate=True,generate_mode='stochastic',temperature=2)
# time.sleep(30)

# In[ ]:




