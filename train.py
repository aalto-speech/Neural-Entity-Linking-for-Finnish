import random
import torch
import torch.nn.functional as F
import numpy as np


def train(pairs_batch_train, pairs_batch_dev, model, model_optimizer, criterion, batch_size, device):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    target = torch.ones(batch_size).to(device)

    for epoch in range(30):
        train_epoch_loss = []
        dev_epoch_loss = []

        model.train()
        
        for iteration, batch in enumerate(pairs_batch_train):
            train_loss = 0        
            model.zero_grad()
 
            context, context_lengths, mention, true_entity, candidate_1, candidate_2 = batch
            context, mention, true_entity, candidate_1, candidate_2 = context.to(device), mention.to(device), true_entity.to(device), candidate_1.to(device), candidate_2.to(device)
                
            context_mention, true_entity, candidate_1, candidate_2 = model(context, context_lengths, mention, true_entity, candidate_1, candidate_2)

            context_true_cos = cos(context_mention, true_entity)
            context_candidate_1_cos = cos(context_mention, candidate_1)
            context_candidate_2_cos = cos(context_mention, candidate_2)

            train_loss += criterion(context_true_cos, context_candidate_1_cos, target) 
            train_loss += criterion(context_true_cos, context_candidate_2_cos, target)

            train_epoch_loss.append(train_loss.item())

            # backward step
            train_loss.backward()
            model_optimizer.step()


        with torch.no_grad():
            model.eval()

            for iteration, batch in enumerate(pairs_batch_dev):
                dev_loss = 0        
 
                context, context_lengths, mention, true_entity, candidate_1, candidate_2 = batch
                context, mention, true_entity, candidate_1, candidate_2 = context.to(device), mention.to(device), true_entity.to(device), candidate_1.to(device), candidate_2.to(device)
                    
                context_mention, true_entity, candidate_1, candidate_2 = model(context, context_lengths, mention, true_entity, candidate_1, candidate_2)

                context_true_cos = cos(context_mention, true_entity)
                context_candidate_1_cos = cos(context_mention, candidate_1)
                context_candidate_2_cos = cos(context_mention, candidate_2)

                dev_loss += criterion(context_true_cos, context_candidate_1_cos, target) 
                dev_loss += criterion(context_true_cos, context_candidate_2_cos, target)

                dev_epoch_loss.append(dev_loss.item())

        
        train_epoch_loss = np.array(train_epoch_loss)
        dev_epoch_loss = np.array(dev_epoch_loss)

        train_epoch_loss = np.mean(train_epoch_loss)
        dev_epoch_loss = np.mean(dev_epoch_loss)

        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, train_epoch_loss, dev_epoch_loss))



        print('saving the models...')
        torch.save({
        'model': model.state_dict(),
        'model_optimizer': model_optimizer.state_dict(),
        }, 'weights/model_margin/state_dict_' + str(epoch+1) + '.pt')

