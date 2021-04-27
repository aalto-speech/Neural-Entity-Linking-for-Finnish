import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class NELModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(NELModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.lin_1 = nn.Linear(self.hidden_size+self.embedding_dim, self.hidden_size)
        self.lin_2 = nn.Linear(self.embedding_dim, self.hidden_size)
        self.lin_1_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.lin_2_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
                            self.embedding_dim,
                            self.hidden_size,
                            num_layers=1,
                            bidirectional=True
                            )

        
        

    def forward(self, context, context_lengths, mention, true_entity, candidate_1, candidate_2, evaluation=False):
        if evaluation == False:
            # process the context and mention
            context = pack_padded_sequence(context, context_lengths)        
            output, hidden = self.lstm(context) 
            output = pad_packed_sequence(output)[0]
            
            hidden = torch.mean(hidden[0], dim=0, keepdim=True)
            mention = torch.mean(mention, dim=0, keepdim=True)
            
            context_mention = torch.cat((hidden, mention), dim=2)
            context_mention = self.lin_1(context_mention)[0]
            context_mention = self.dropout(context_mention)
            
            context_mention = self.relu(context_mention)
            context_mention = self.lin_1_1(context_mention)
            context_mention = self.dropout(context_mention)

            # process the candidates
            true_entity = torch.mean(true_entity, dim=0, keepdim=True)
            candidate_1 = torch.mean(candidate_1, dim=0, keepdim=True)
            candidate_2 = torch.mean(candidate_2, dim=0, keepdim=True)
            
            true_entity = self.lin_2(true_entity)[0]
            true_entity = self.dropout(true_entity)
            true_entity = self.relu(true_entity)
            true_entity = self.lin_2_2(true_entity)
            true_entity = self.dropout(true_entity)

            candidate_1 = self.lin_2(candidate_1)[0]
            candidate_1 = self.dropout(candidate_1)
            candidate_1 = self.relu(candidate_1)
            candidate_1 = self.lin_2_2(candidate_1)
            candidate_1 = self.dropout(candidate_1)

            candidate_2 = self.lin_2(candidate_2)[0]
            candidate_2 = self.dropout(candidate_2)
            candidate_2 = self.relu(candidate_2)
            candidate_2 = self.lin_2_2(candidate_2)
            candidate_2 = self.dropout(candidate_2)

            return context_mention, true_entity, candidate_1, candidate_2


        elif evaluation == True:
            # process the context and mention
            # in evaluation mode, the true_entity is the first candidate
            context = context.permute(1, 0, 2)
            mention = mention.permute(1, 0, 2)

            output, hidden = self.lstm(context)
            hidden = torch.mean(hidden[0], dim=0, keepdim=True)
            
            mention = torch.mean(mention, dim=0, keepdim=True)
            
            context_mention = torch.cat((hidden, mention), dim=2)
            context_mention = self.lin_1(context_mention)[0]
            context_mention = self.dropout(context_mention)
           
            context_mention = self.relu(context_mention)
            context_mention = self.lin_1_1(context_mention)
            context_mention = self.dropout(context_mention)


            # process the candidates
            if candidate_2 != '':
                true_entity = true_entity.permute(1, 0, 2)
                candidate_1 = candidate_1.permute(1, 0, 2)
                candidate_2 = candidate_2.permute(1, 0, 2)
                
                true_entity = torch.mean(true_entity, dim=0, keepdim=True)
                candidate_1 = torch.mean(candidate_1, dim=0, keepdim=True)
                candidate_2 = torch.mean(candidate_2, dim=0, keepdim=True)
                 
                true_entity = self.lin_2(true_entity)[0]
                true_entity = self.dropout(true_entity)
                true_entity = self.relu(true_entity)
                true_entity = self.lin_2_2(true_entity)
                true_entity = self.dropout(true_entity)
                
                candidate_1 = self.lin_2(candidate_1)[0]
                candidate_1 = self.dropout(candidate_1)
                candidate_1 = self.relu(candidate_1)
                candidate_1 = self.lin_2_2(candidate_1)
                candidate_1 = self.dropout(candidate_1)
                
                candidate_2 = self.lin_2(candidate_2)[0]
                candidate_2 = self.dropout(candidate_2)
                candidate_2 = self.relu(candidate_2)
                candidate_2 = self.lin_2_2(candidate_2)
                candidate_2 = self.dropout(candidate_2)

                
            
            elif candidate_1 != '' and candidate_2 == '':
                true_entity = true_entity.permute(1, 0, 2)
                candidate_1 = candidate_1.permute(1, 0, 2)

                true_entity = torch.mean(true_entity, dim=0, keepdim=True)
                candidate_1 = torch.mean(candidate_1, dim=0, keepdim=True)
                
                true_entity = self.lin_2(true_entity)[0]
                true_entity = self.dropout(true_entity)
                true_entity = self.relu(true_entity)
                true_entity = self.lin_2_2(true_entity)
                true_entity = self.dropout(true_entity)
                
                candidate_1 = self.lin_2(candidate_1)[0]
                candidate_1 = self.dropout(candidate_1)
                candidate_1 = self.relu(candidate_1)
                candidate_1 = self.lin_2_2(candidate_1)
                candidate_1 = self.dropout(candidate_1)

                candidate_2 = ''
            
            
            elif candidate_1 == '' and candidate_2 == '':
                true_entity = true_entity.permute(1, 0, 2)

                true_entity = torch.mean(true_entity, dim=0, keepdim=True)
                
                true_entity = self.lin_2(true_entity)[0]
                true_entity = self.dropout(true_entity)
                true_entity = self.relu(true_entity)
                true_entity = self.lin_2_2(true_entity)
                true_entity = self.dropout(true_entity)
                
               
                candidate_1 = ''
                candidate_2 = ''


 
            return context_mention, true_entity, candidate_1, candidate_2


