import numpy as np
import os
import requests

import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

from pywikibot.data import api
import pywikibot
from pywikibot import pagegenerators as pg
import json
from wikidata.client import Client
import random
import pprint
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import re
import fasttext
from sklearn.metrics import f1_score, precision_score, recall_score
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from wikipedia2vec import Wikipedia2Vec
from uralicNLP import uralicApi

from model import NELModel
from config.config import *
import utils.prepare_data as prepare_data


def get_name(ID):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    
    sparql.setQuery("""
                    SELECT *
                    WHERE
                    {
                        wd:""" + ID + """ rdfs:label ?label.
                        FILTER (langMatches( lang(?label), "FI" ) )

                    }
                    LIMIT 1
                    """)


    sparql.setReturnFormat(JSON)
    query = sparql.query().convert()
    results = query['results']
    bindings = results['bindings']
    try:
        label = bindings[0]['label']
        entity = label['value']
    except:
        entity = ''
    
    return entity
            

def get_description(ID):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    
    # get description
    sparql.setQuery("""
                    SELECT *
                    WHERE
                    {
                        wd:""" + ID + """ schema:description ?o.
                        FILTER ( lang(?o) = "fi" )
                    }
                    LIMIT 1
                    """)

    sparql.setReturnFormat(JSON)
    query = sparql.query().convert()
    results = query['results']
    bindings = results['bindings']
    if len(bindings) != 0:
        label = bindings[0]['o']
        description = label['value']
    else:
        description = ' '

    return description


def get_id(mention):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    split_mention = mention.split()
    lemmatized_mention = [uralicApi.lemmatize(word, "fin", word_boundaries=False) for word in split_mention]

    mention_merged = []
    for word in lemmatized_mention:
        if len(word) == 1:
            mention_merged.append(word[0])
        elif len(word) == 2:
            if len(word[0]) <= len(word[1]):
                mention_merged.append(word[0])
            else:
                mention_merged.append(word[1])
        else:
            pass
    
    if len(mention_merged) != 0:
        mention_merged = ' '.join(mention_merged)
        mention = mention_merged
    
    items = getItems(site, mention)
    ent_id = []
    alias = []
    if len(items['search']) == 0:
        ent_id = []
        alias = []
    else:
        items = items['search']
        for entry in items:
            try:
                ent_id.append(entry['id'])
            except:
                ent_id.append(' ')
            try:
                alias.append(entry['match']['text'])
            except:
                alias.append(' ')
    
    return ent_id, alias



def getItems(site, itemtitle):
    params = { 'action' :'wbsearchentities' , 'format' : 'json' , 'language' : 'fi', 'type' : 'item', 'search': itemtitle, 'limit': 3}
    request = api.Request(site=site,**params)
    return request.submit()


def prepare_word_sequence(seq, embeddings, entity_embeddings,  is_entity):
    res = []
    seq = seq.split()
    for w in seq:
        try:
            if is_entity == True:
                res.append(entity_embeddings.get_entity_vector(w))
            else:
                res.append(entity_embeddings.get_word_vector(w))
        except:
            res.append(embeddings[w])
    
    res = autograd.Variable(torch.FloatTensor(res))

    return res



def word_to_idx(data, embeddings, entity_embeddings, is_entity=False):
    res = []
    res.append(prepare_word_sequence(data, embeddings, entity_embeddings, is_entity))

    return res


def cat_tensors(candidate_name, candidate_alias, candidate_desc):
    if candidate_name.size(1) != 0 and candidate_alias.size(1) != 0 and candidate_desc.size(1) != 0:
        res = torch.cat((candidate_name, candidate_alias, candidate_desc), dim=1)
    
    elif candidate_name.size(1) != 0 and candidate_alias.size(1) != 0 and candidate_desc.size(1) == 0:
        res = torch.cat((candidate_name, candidate_alias), dim=1)
    
    elif candidate_name.size(1) != 0 and candidate_alias.size(1) == 0 and candidate_desc.size(1) != 0:
        res = torch.cat((candidate_name, candidate_desc), dim=1)

    elif candidate_name.size(1) != 0 and candidate_alias.size(1) == 0 and candidate_desc.size(1) == 0:
        res = candidate_name
    
    elif candidate_name.size(1) == 0 and candidate_alias.size(1) != 0 and candidate_desc.size(1) != 0:
        res = torch.cat((candidate_alias, candidate_desc), dim=1)

    elif candidate_name.size(1) == 0 and candidate_alias.size(1) == 0 and candidate_desc.size(1) != 0:
        res = candidate_desc

    elif candidate_name.size(1) == 0 and candidate_alias.size(1) != 0 and candidate_desc.size(1) == 0:
        res = candidate_alias
    else:
        res = torch.zeros(1, 1, 300).to(device)

    return res




# ---------------------------------------------------------------------------

# combine compound entities in one line, remove the ASR errors and remove <DOCUMENT> and doc_id
def format_data(test_path, formatted_path):
    with open(test_path, 'r') as f:
        data = f.readlines()

    with open(formatted_path, 'w') as f:
        for i in range(len(data)):
            compound_entities = []
            if data[i] != '\n':
                if '<DOCUMENT>' not in data[i] and 'doc_id' not in data[i]:
                    line = data[i].split('\t')
                    word = line[0]
                    tag = line[1].rstrip()
                    
                    try:
                        if 'ASR' in line[2]:
                            asr_error = True
                        else:
                            asr_error = False
                    except:
                            asr_error = False
                    
                    
                    if asr_error == False or asr_error == True:
                        if 'B-' in tag:
                            compound_entities.append(word)
                            tag_to_save = tag
                            tag_to_save = tag_to_save.replace('B-', '').replace('I-', '')

                            for j in range(i+1, len(data)):
                                if data[j] != '\n':
                                    word = data[j].split('\t')[0]
                                    tag = data[j].split('\t')[1]
                                    if 'I-' in tag:
                                        compound_entities.append(word)
                                    else:
                                        break
                                else:
                                    break
               
                            compound_entities = ' '.join(compound_entities)
                            f.write(compound_entities + '\t' + tag_to_save + '\n')
                            compound_entities = []
               
                        elif tag == 'O':
                            f.write(word + '\t' + tag + '\n')

            else:
                f.write('\n')



# remove sentences that do not have links to Wikidata
def remove_non_links(formatted_data_path, test_preprocessed_path):
    with open(formatted_data_path, 'r') as f:
        data = f.readlines()

    with open(test_preprocessed_path, 'w') as f:

        sentence = []
        entities = []
        for line in data:
            if line != '\n':
                if '<DOCUMENT>' not in line and 'doc_id' not in line: 
                    sentence.append(line)
                    tag = line.split('\t')[1].rstrip()
                    
                    if tag != 'O':
                        entities.append(tag)
            
            else:
                if len(entities) != 0:
                    for word in sentence:
                        f.write(word)
                    f.write('\n')
                    sentence = []
                    entities = []


# remove sentences that do not have links to Wikidata but also splits the words if they end with a punctuation
#def remove_non_links(formatted_data_path, test_preprocessed_path):
#    import string
#
#    with open(formatted_data_path, 'r') as f:
#        data = f.readlines()
#
#    with open(test_preprocessed_path, 'w') as f:
#
#        sentence = []
#        entities = []
#        for line in data:
#            if line != '\n':
#                if '<DOCUMENT>' not in line and 'doc_id' not in line:
#                    word = line.split('\t')[0]
#                    tag = line.split('\t')[1].rstrip()
#
#                    if word[-1] in string.punctuation:
#                        sentence.append(word[:-1] + '\t' + tag + '\n')
#                        sentence.append(word[-1] + '\t' + tag + '\n')
#                    else:
#                        sentence.append(line)
#                                        
#                    if tag != 'O':
#                        entities.append(tag)
#            
#            else:
#                if len(entities) != 0:
#                    for word in sentence:
#                        f.write(word)
#                    f.write('\n')
#                    sentence = []
#                    entities = []
#

 
# returns all the contexts and mentions from the data
def get_context_mention(test_preprocessed_path):
    with open(test_preprocessed_path, 'r') as f:
        data = f.readlines()
    
    sentence = []
    entities = []
    contexts = []
    mentions = []

    for line in data:
        if line != '\n':
            word = line.split('\t')[0].rstrip()
            tag = line.split('\t')[1].rstrip()
            
            if tag != 'O': 
                entities.append(word)
            sentence.append(word)

        else:
            sentence = ' '.join(sentence)
            
            if len(entities) == 0:
                mentions.append('')
                contexts.append(sentence)
            else:
                for ent in entities:
                    contexts.append(sentence.replace(ent, '', 1))
                    mentions.append(ent)

            sentence = []
            entities = []
    
    return contexts, mentions



# generates predictions and saves them to a numpy array
def cache_predictions(contexts, mentions, save_path, model, embeddings, entity_embeddings, cos):
    predictions = []
    for i in range(len(contexts)):
        print(i) 
        if len(mentions[i]) == 0:
            pass
        else:
            with open('cached_results_asr.txt', 'a') as f:

                candidates_id, alias = get_id(mentions[i])
                # if there are not candidates returned, then the prediction is 'NIL'
                if len(candidates_id) == 0:
                    # CACHING                    
                    f.write('NIL' + '\n')

                # there are some candidates returned
                else:
                    # CACHING
                    string_line = []
                    string_line.append(contexts[i])
                    string_line.append(mentions[i])

                    if len(candidates_id) == 3:
                        candidate_1_name = get_name(candidates_id[0])
                        candidate_1_desc = get_description(candidates_id[0])
                        candidate_1_alias = alias[0]

                        candidate_2_name = get_name(candidates_id[1])
                        candidate_2_desc = get_description(candidates_id[1])
                        candidate_2_alias = alias[1]

                        candidate_3_name = get_name(candidates_id[2])
                        candidate_3_desc = get_description(candidates_id[2])
                        candidate_3_alias = alias[2]
                        
                        # CACHING
                        string_line.append(candidates_id[0])
                        string_line.append(candidate_1_name)
                        string_line.append(candidate_1_desc)
                        string_line.append(candidate_1_alias)

                        string_line.append(candidates_id[1])
                        string_line.append(candidate_2_name)
                        string_line.append(candidate_2_desc)
                        string_line.append(candidate_2_alias)

                        string_line.append(candidates_id[2])
                        string_line.append(candidate_3_name)
                        string_line.append(candidate_3_desc)
                        string_line.append(candidate_3_alias)


                    elif len(candidates_id) == 2:
                        candidate_1_name = get_name(candidates_id[0])
                        candidate_1_desc = get_description(candidates_id[0])
                        candidate_1_alias = alias[0]
                        #candidate_1 = candidate_1_name + ' ' + candidate_1_desc + ' ' + candidate_1_alias

                        candidate_2_name = get_name(candidates_id[1])
                        candidate_2_desc = get_description(candidates_id[1])
                        candidate_2_alias = alias[1]
                        #candidate_2 = candidate_2_name + ' ' + candidate_2_desc + ' ' + candidate_2_alias

                        candidate_3 = ''

                        # CACHING
                        string_line.append(candidates_id[0])
                        string_line.append(candidate_1_name)
                        string_line.append(candidate_1_desc)
                        string_line.append(candidate_1_alias)

                        string_line.append(candidates_id[1])
                        string_line.append(candidate_2_name)
                        string_line.append(candidate_2_desc)
                        string_line.append(candidate_2_alias)


                    elif len(candidates_id) == 1:
                        candidate_1_name = get_name(candidates_id[0])
                        candidate_1_desc = get_description(candidates_id[0])
                        candidate_1_alias = alias[0]
                        #candidate_1 = candidate_1_name + ' ' + candidate_1_desc + ' ' + candidate_1_alias
                        
                        candidate_2 = ''
                        candidate_3 = ''
                       

                        # CACHING
                        string_line.append(candidates_id[0])
                        string_line.append(candidate_1_name)
                        string_line.append(candidate_1_desc)
                        string_line.append(candidate_1_alias)

                  

                    # CACHING
                    string_line = '\t'.join(string_line)
                    f.write(string_line + '\n')
                    string_line = []
                 


# generates predictions and saves them to a numpy array
def save_predictions(contexts, mentions, save_path, model, embeddings, entity_embeddings, cos):
    predictions = []
    with open('cached_results_asr.txt', 'r') as f:
        data = f.readlines()
    
    context_number = 0
    number_nil = 0
    for i in range(len(contexts)):
        if len(mentions[i]) == 0:
            number_nil += 1
            pass
    
        # get cached candidates
        else:
            line = data[context_number]
            context_number += 1
            line = line.split('\t')
            line = [l.rstrip() for l in line]

            candidates_id = []

            try:
                candidates_id.append(line[2])
            except:
                pass
       
            try:
                candidates_id.append(line[6])
            except:
                pass

            try:
                candidates_id.append(line[10])
            except:
                pass



            if len(candidates_id) == 0:
                predictions.append('O')
            # there are some candidates returned
            else:
                context = contexts[i]
                mention = mentions[i]
                
                # index the context and mention
                if len(context) == 0:
                    context = torch.zeros(1, 1, 300).to(device)
                else:
                    context = word_to_idx(context, embeddings, entity_embeddings)
                    context = torch.stack(context).to(device)

                mention = word_to_idx(mention, embeddings, entity_embeddings, is_entity=True)
                mention = torch.stack(mention).to(device)

                if len(candidates_id) == 3:
                    candidate_1_name = line[3]
                    candidate_1_desc = line[4]
                    candidate_1_alias = line[5]

                    candidate_2_name = line[7]
                    candidate_2_desc = line[8]
                    candidate_2_alias = line[9]

                    candidate_3_name = line[11]
                    candidate_3_desc = line[12]
                    candidate_3_alias = line[13]



                    # index candidates
                    # candidate 1
                    candidate_1_name = word_to_idx(candidate_1_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_name = torch.stack(candidate_1_name).to(device)

                    candidate_1_alias = word_to_idx(candidate_1_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_alias = torch.stack(candidate_1_alias).to(device)
                    
                    candidate_1_desc = word_to_idx(candidate_1_desc, embeddings, entity_embeddings)
                    candidate_1_desc = torch.stack(candidate_1_desc).to(device)

                    candidate_1 = cat_tensors(candidate_1_name, candidate_1_alias, candidate_1_desc)

                    # candidate 2
                    candidate_2_name = word_to_idx(candidate_2_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_2_name = torch.stack(candidate_2_name).to(device)

                    candidate_2_alias = word_to_idx(candidate_2_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_2_alias = torch.stack(candidate_2_alias).to(device)
                    
                    candidate_2_desc = word_to_idx(candidate_2_desc, embeddings, entity_embeddings)
                    candidate_2_desc = torch.stack(candidate_2_desc).to(device)
                    
                    candidate_2 = cat_tensors(candidate_2_name, candidate_2_alias, candidate_2_desc)

                    # candidate 3
                    candidate_3_name = word_to_idx(candidate_3_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_3_name = torch.stack(candidate_3_name).to(device)

                    candidate_3_alias = word_to_idx(candidate_3_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_3_alias = torch.stack(candidate_3_alias).to(device)
                    
                    candidate_3_desc = word_to_idx(candidate_3_desc, embeddings, entity_embeddings)
                    candidate_3_desc = torch.stack(candidate_3_desc).to(device)
                    
                    candidate_3 = cat_tensors(candidate_3_name, candidate_3_alias, candidate_3_desc)


                elif len(candidates_id) == 2:
                    candidate_1_name = line[3]
                    candidate_1_desc = line[4]
                    candidate_1_alias = line[5]

                    candidate_2_name = line[7]
                    candidate_2_desc = line[8]
                    candidate_2_alias = line[9]

                    candidate_3 = ''


                    # index candidates
                    # candidate 1
                    candidate_1_name = word_to_idx(candidate_1_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_name = torch.stack(candidate_1_name).to(device)

                    candidate_1_alias = word_to_idx(candidate_1_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_alias = torch.stack(candidate_1_alias).to(device)
                    
                    candidate_1_desc = word_to_idx(candidate_1_desc, embeddings, entity_embeddings)
                    candidate_1_desc = torch.stack(candidate_1_desc).to(device)

                    candidate_1 = cat_tensors(candidate_1_name, candidate_1_alias, candidate_1_desc)

                    # candidate 2
                    candidate_2_name = word_to_idx(candidate_2_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_2_name = torch.stack(candidate_2_name).to(device)

                    candidate_2_alias = word_to_idx(candidate_2_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_2_alias = torch.stack(candidate_2_alias).to(device)
                    
                    candidate_2_desc = word_to_idx(candidate_2_desc, embeddings, entity_embeddings)
                    candidate_2_desc = torch.stack(candidate_2_desc).to(device)

                    candidate_2 = cat_tensors(candidate_2_name, candidate_2_alias, candidate_2_desc)



                elif len(candidates_id) == 1:
                    candidate_1_name = line[3]
                    candidate_1_desc = line[4]
                    candidate_1_alias = line[5]

                    candidate_2 = ''
                    candidate_3 = ''
                   

                    # index candidates
                    # candidate 1
                    candidate_1_name = word_to_idx(candidate_1_name, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_name = torch.stack(candidate_1_name).to(device)

                    candidate_1_alias = word_to_idx(candidate_1_alias, embeddings, entity_embeddings, is_entity=True)
                    candidate_1_alias = torch.stack(candidate_1_alias).to(device)
                    
                    candidate_1_desc = word_to_idx(candidate_1_desc, embeddings, entity_embeddings)
                    candidate_1_desc = torch.stack(candidate_1_desc).to(device)

                    candidate_1 = cat_tensors(candidate_1_name, candidate_1_alias, candidate_1_desc)
               

                
                # forward pass
                context_lengths = 0
                context_mention, candidate_1, candidate_2, candidate_3 = model(context, context_lengths, mention, candidate_1, candidate_2, candidate_3, evaluation=True)

                # get similarity scores
                scores = []
                
                context_candidate_1_cos = cos(context_mention, candidate_1)
                scores.append(context_candidate_1_cos.item())

                if candidate_2 != '':
                    context_candidate_2_cos = cos(context_mention, candidate_2)
                    scores.append(context_candidate_2_cos.item())

                if candidate_3 != '':
                    context_candidate_3_cos = cos(context_mention, candidate_3)
                    scores.append(context_candidate_3_cos.item())


                # get the item with max score and store it to predictions
                predicted_index = scores.index(max(scores))
                if scores[predicted_index] >= 0.1:
                    predictions.append(candidates_id[predicted_index])
                else:
                    predictions.append('O')
    

    print(number_nil)

    # save the predictions
    predictions = np.array(predictions)
    np.save(save_path, predictions)
     



def get_predictions_in_batch(contexts, mentions, save_path, model, embeddings, entity_embeddings, cos):
    #ranges = [(0, 30), (30, 60), (60, 90), (90, 150), (150, 180), (180, 210), (210, 230), (230, 260)]
    ranges = [(210, 240)]
    for i in ranges:
        print(i)
        successful = False
        contexts_batch = contexts[i[0]:i[1]]
        mentions_batch = mentions[i[0]:i[1]]
        
        cache_predictions(contexts_batch, mentions_batch, save_path, model, embeddings, entity_embeddings, cos)

        #while successful == False:
        #    time.sleep(5)
        #    try:
        #        cache_predictions(contexts_batch, mentions_batch, save_path, model, embeddings, entity_embeddings, cos)
        #    except:
        #        pass
        


# save the predictions to a txt file in a IOB format
def save_predictions_to_txt(predictions_path, test_preprocessed_path, predictions):
    #with open(test_preprocessed_path, 'r') as f:
    with open('data/urheiluruutu/formatted_test_asr.txt', 'r') as f:
        data = f.readlines()
    
    with open(predictions_path, 'w') as f:
        pred_idx = 0
        for line in data:
            if line != '\n':
                line = line.split('\t')
                word = line[0]
                tag = line[1].rstrip()
                
                if tag != 'O':
                    #print(pred_idx)
                    predicted_tag = predictions[pred_idx]
                    pred_idx += 1
                    for word in word.split():
                        f.write(word + '\t' + predicted_tag + '\n')
                else:
                    f.write(word + '\t' + tag + '\n')

            else:
                f.write('\n')



# evaluate the system
def evaluate(test_preprocessed_path, predictions_path):
    # get true labels
    with open(test_preprocessed_path, 'r') as f:
        data = f.readlines()

    true = []
    for line in data:
        if line != '\n':
            line = line.split('\t')
            tag = line[1].rstrip().replace('B-', '').replace('I-', '')
            
            #if tag != 'O':
            true.append(tag)
    
    
    # get predicted labels
    with open(predictions_path, 'r') as f:
        data = f.readlines()

    predictions = []
    for line in data:
        if line != '\n':
            line = line.split('\t')
            tag = line[1].rstrip()
            
            #if tag != 'O':
            predictions.append(tag)
    

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(true)):
        if true[i] == 'O' and predictions[i] == 'O':
            pass
        else:
            if true[i] == predictions[i]:
                tp += 1
            elif true[i] != 'O' and predictions[i] == 'O':
                fn += 1
            elif (true[i] != 'O' and predictions[i] != 'O'):
                fp += 1
            elif true[i] == 'O' and predictions[i] != 'O':
                fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * (precision * recall)) / (precision + recall)
    

    # replace tags and predictions with indices
    #combined = true + predictions
    #tag2idx = {}
    #
    #for tag in combined:
    #    if tag not in tag2idx.keys():
    #        tag2idx[tag] = len(tag2idx)


    #indexed_true = []
    #indexed_predictions = []
    #for i in range(len(true)):
    #    indexed_true.append(tag2idx[true[i]])
    #    indexed_predictions.append(tag2idx[predictions[i]])

   
    #print('Micro Precision: %f' % (precision_score(true, predictions, average='micro')))
    #print('Micro Recall: %f' % (recall_score(true, predictions, average='micro')))
    #print('Micro F1: %f' % (f1_score(true, predictions, average='micro')))
    
    print('Precision: %f' % (precision))
    print('Recall: %f' % (recall))
    print('F1: %f' % (f1))


    
if __name__ == '__main__':
    
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    print('Loading embeddings...')
    embeddings = fasttext.load_model('weights/embeddings/cc.fi.300.bin')
    entity_embeddings = Wikipedia2Vec.load('weights/embeddings/entity_embeddings')
    print('Done...')
   

    # Login to wikidata
    site = pywikibot.Site("wikidata", "wikidata")
    repo = site.data_repository()
    
    test_data_path = 'data/urheiluruutu/urheiluruutu_test_ner_asr.txt'
    formatted_data_path = 'data/urheiluruutu/formatted_test_asr.txt'
    test_preprocessed_path = 'data/urheiluruutu/formatted_test_gold_asr.txt'
    predictions_path = 'data/urheiluruutu/predictions_test_asr.txt'
    save_path =  'predictions/margin/predictions_23.npy'
    
    #format_data(test_data_path, formatted_data_path)

    # used to prepare the gold set. Usually run only once
    #format_data('data/urheiluruutu/urheiluruutu_test.txt', 'data/urheiluruutu/formatted_test_gold_asr.txt')

    # Login to wikidata
    site = pywikibot.Site("wikidata", "wikidata")
    repo = site.data_repository()
    
    contexts, mentions = get_context_mention(formatted_data_path)

    # initialize and load the NELModel
    model = NELModel(embedding_dim, hidden_size).to(device)
    checkpoint = torch.load('weights/model/state_dict_23.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # processes the data in batches and caches the retrieved candidates
    #get_predictions_in_batch(contexts, mentions, save_path, model, embeddings, entity_embeddings, cos)
    
    save_predictions(contexts, mentions, save_path, model, embeddings, entity_embeddings, cos)
    
    predictions = np.load(save_path)
    save_predictions_to_txt(predictions_path, test_preprocessed_path, predictions)
    evaluate(test_preprocessed_path, predictions_path)



