import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import fasttext
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from wikipedia2vec import Wikipedia2Vec

from model import NELModel
from train import train
import utils.prepare_data as prepare_data
from config.config import *


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load features and labels
print('Loading data..')

context_train, mention_train, true_entity_train, candidate_1_train, candidate_2_train = prepare_data.load_data('data/urheiluruutu/urheiluruutu_formatted_whole.txt')
context_dev, mention_dev, true_entity_dev, candidate_1_dev, candidate_2_dev = prepare_data.load_data('data/urheiluruutu/urheiluruutu_formatted_whole.txt')


print('Done...')

print('Loading embeddings...')
embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
entity_embeddings = Wikipedia2Vec.load('weights/embeddings/entity_embeddings')


context_train = prepare_data.word_to_idx(context_train, embeddings, entity_embeddings)
mention_train = prepare_data.word_to_idx(mention_train, embeddings, entity_embeddings, is_entity=True)
true_entity_train = prepare_data.word_to_idx(true_entity_train, embeddings, entity_embeddings, is_candidate=True)
candidate_1_train = prepare_data.word_to_idx(candidate_1_train, embeddings, entity_embeddings, is_candidate=True)
candidate_2_train = prepare_data.word_to_idx(candidate_2_train, embeddings, entity_embeddings, is_candidate=True)

context_dev = prepare_data.word_to_idx(context_dev, embeddings, entity_embeddings)
mention_dev = prepare_data.word_to_idx(mention_dev, embeddings, entity_embeddings, is_entity=True)
true_entity_dev = prepare_data.word_to_idx(true_entity_dev, embeddings, entity_embeddings, is_candidate=True)
candidate_1_dev = prepare_data.word_to_idx(candidate_1_dev, embeddings, entity_embeddings, is_candidate=True)
candidate_2_dev = prepare_data.word_to_idx(candidate_2_dev, embeddings, entity_embeddings, is_candidate=True)


# combine data
train_data = prepare_data.combine_data(context_train, mention_train, true_entity_train, candidate_1_train, candidate_2_train)
dev_data = prepare_data.combine_data(context_dev, mention_dev, true_entity_dev, candidate_1_dev, candidate_2_dev)


# remove extra data that does not fit in the batch
train_data = prepare_data.remove_extra(train_data, batch_size)
dev_data = prepare_data.remove_extra(dev_data, batch_size)


pairs_batch_train = DataLoader(dataset=train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)

pairs_batch_dev = DataLoader(dataset=dev_data,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=prepare_data.collate,
                    pin_memory=True)


# initialize the NELModel
model = NELModel(embedding_dim, hidden_size).to(device)
model_optimizer = optim.Adam(model.parameters(), lr=lr)


# train
if skip_training == False:
    criterion = nn.MarginRankingLoss(margin=1)
    train(pairs_batch_train, pairs_batch_dev, model, model_optimizer, criterion, batch_size, device) 

else:
    checkpoint = torch.load('weights/model/state_dict_1.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])


