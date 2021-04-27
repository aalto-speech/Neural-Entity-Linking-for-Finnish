import numpy as np
import os

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence


def load_data(path): 
    context = []
    mention = []
    true_entity = []
    candidate_1 = []
    candidate_2 = []

    with open(path, 'r') as f:
        data = f.readlines()

    for line in data:
        line = line.split('\t')
        context.append(line[0])
        mention.append(line[1])
        true_entity.append(line[2].split('~'))
        candidate_1.append(line[3].split('~'))
        candidate_2.append(line[4].split('~'))
    
    
    return context, mention, true_entity, candidate_1, candidate_2 



def prepare_word_sequence(seq, embeddings, entity_embeddings, is_entity):
    res = []
    if is_entity == False:
        seq = seq.split()
        for w in seq:
            try:
                res.append(entity_embeddings.get_word_vector(w))
            except:
                res.append(embeddings[w])
        
    else:
        try:
            res.append(entity_embeddings.get_entity_vector(seq))
        except:
            seq = seq.split()
            for w in seq:
                try:
                    res.append(entity_embeddings.get_word_vector(w))
                except:
                    res.append(embeddings[w])

    res = autograd.Variable(torch.FloatTensor(res))

    return res


def prepare_candidates(seq, embeddings, entity_embeddings):
    res = []
    name = seq[0]
    desc = seq[1]
    alias = seq[2]
    
    # get name embedding
    if name != '':
        try:
            res.append(entity_embeddings.get_entity_vector(name))
        except:
            name = name.split()
            for w in name:
                try:
                    res.append(entity_embeddings.get_word_vector(w))
                except:
                    #res.append(np.random.normal(scale=0.6, size=(300, )))
                    res.append(embeddings[w])

    
    # get alias embedding
    if alias != '':
        try:
            res.append(entity_embeddings.get_word_vector(alias))
        except:
            alias = alias.split()
            for w in alias:
                try:
                    res.append(entity_embeddings.get_word_vector(w))
                except:
                    #res.append(np.random.normal(scale=0.6, size=(300, )))
                    res.append(embeddings[w])


    # get description embedding
    if desc != '':
        desc = desc.split()
        for w in desc:
            try:
                res.append(entity_embeddings.get_word_vector(w))
            except:
                #res.append(np.random.normal(scale=0.6, size=(300, )))
                res.append(embeddings[w])

    res = autograd.Variable(torch.FloatTensor(res))

    return res




def word_to_idx(data, embeddings, entity_embeddings, is_entity=False, is_candidate=False):
    res = []
    if is_candidate == False:
        for line in data:
            res.append(prepare_word_sequence(line, embeddings, entity_embeddings, is_entity))
    else:
        for line in data:
            res.append(prepare_candidates(line, embeddings, entity_embeddings))

    return res


def combine_data(indexed_context, indexed_mention, indexed_true_entity, indexed_candidate_1, indexed_candidate_2):
    res = []
    
    for i in range(len(indexed_context)):
        res.append((indexed_context[i], indexed_mention[i], indexed_true_entity[i], indexed_candidate_1[i], indexed_candidate_2[i]))

    return res


def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    if extra != 0:
        data = data[:-extra][:]

    return data


def collate(list_of_samples):
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)
    
    context, mention, true_entity, candidate_1, candidate_2 = zip(*list_of_samples)
    context_lengths = [len(seq) for seq in context]


    padding_value = 0

    # pad context
    pad_context = pad_sequence(context, padding_value=padding_value)
   
    # pad mention
    pad_mention = pad_sequence(mention, padding_value=padding_value)

    # pad true_entity
    pad_true_entity = pad_sequence(true_entity, padding_value=padding_value)
    
    # pad candidate_1
    pad_candidate_1 = pad_sequence(candidate_1, padding_value=padding_value)
    
    # pad candidate_2
    pad_candidate_2 = pad_sequence(candidate_2, padding_value=padding_value)


    return pad_context, context_lengths, pad_mention, pad_true_entity, pad_candidate_1, pad_candidate_2








