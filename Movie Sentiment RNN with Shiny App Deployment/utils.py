# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:09:22 2019
Author: Francisco Javier Carrera Arias
Utils
"""
import numpy as np
import torch.nn as nn
import nltk
import re
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from LSTM_Model import LSTM_Model

# Functions
def pad_features(reviews_ints, seq_length):
    # Returns features of review_ints, where each review is padded with 0's 
    # or truncated to the input seq_length.
    features=[]
    for k in range(len(reviews_ints)):
        if len(reviews_ints[k]) > seq_length:
            features.append(reviews_ints[k][:seq_length])
        elif len(reviews_ints[k]) < seq_length:
            zeros = seq_length - len(reviews_ints[k])
            zero_fill = [0]*zeros
            features.append(zero_fill+reviews_ints[k])
        else:
            features.append(reviews_ints[k])
    
    return np.array(features)

def prune_review_text(reviews, single_rev = False, download_stopwords = False):
    # Clean the review text
    if download_stopwords == True:
        nltk.download('stopwords')
    # Gather stopwords and Initializer stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    if single_rev == False:
         # Text to lowercase
         reviews_lower = [review.lower() for review in reviews]
         # Remove HTML tags
         reviews_noHTML = [re.sub(r"<.*?>"," ",review) for review in reviews_lower]
         # Remove punctuation
         reviews_noPunct = [re.sub(r"[^\w\s]"," ",review) for review in reviews_noHTML]
         # Remove extra whitespace
         reviews_noXtraSpace = [" ".join(re.split("\s+", review)) for review in reviews_noPunct]
         # Tokenize
         tokenized_reviews = [str.split(review) for review in reviews_noXtraSpace]
         # Remove stop words
         tokenized_reviews_2 = [[word for word in review if word not in stop_words] for review in tokenized_reviews]
         # Stemming
         clean_reviews = [[stemmer.stem(word) for word in review] for review in tokenized_reviews_2]
    else:
         # Text to lowercase
         reviews_lower = reviews.lower()
         # Remove HTML tags
         reviews_noHTML = re.sub(r"<.*?>"," ",reviews_lower)
         # Remove punctuation
         reviews_noPunct = re.sub(r"[^\w\s]"," ",reviews_noHTML)
         # Remove extra whitespace
         reviews_noXtraSpace = " ".join(re.split("\s+", reviews_noPunct))
         # Tokenize
         tokenized_reviews = str.split(reviews_noXtraSpace)
         # Remove stop words
         tokenized_reviews_2 = [word for word in tokenized_reviews if word not in stop_words]
         # Stemming
         clean_reviews = [stemmer.stem(word) for word in tokenized_reviews_2]
    return clean_reviews

def train_LSTM_Net(LSTM, train_loader, valid_loader, batch_size,
                   criterion, optimizer, epochs, train_on_gpu, print_every = 100, clip = 5):
    counter = 0
    valLossTracker = []
    # move model to GPU, if available
    if(train_on_gpu):
        LSTM.cuda()
    LSTM.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = LSTM.init_hidden(batch_size)
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            LSTM.zero_grad()
            # get the output from the model
            output, h = LSTM(inputs.long(), h)
            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # gradient clipping helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(LSTM.parameters(), clip)
            optimizer.step()
            if counter % print_every == 0:
                val_h = LSTM.init_hidden(batch_size)
                val_loss = 0
                # Switch LSTM for eval to test on validation set
                LSTM.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        # Creating new variables for the hidden state
                        val_h = tuple([each.data for each in val_h])
                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()
                        output, val_h = LSTM(inputs.long(), val_h)
                        vloss = criterion(output.squeeze(), labels.float())
                        val_loss += vloss.item()
                LSTM.train()
                # Print Metrics
                print("Epoch: {}/{}...".format(e+1, epochs),
                "Step: {}...".format(counter),
                "Val Loss: {:.6f}".format(val_loss/len(valid_loader)))
                valLossTracker.append(val_loss/len(valid_loader))
                # If validation loss decreases, save checkpoint
                if (val_loss/len(valid_loader)) == min(valLossTracker):
                    print("Loss Decreased...Saving Checkpoint")
                    torch.save(LSTM.state_dict(), "Sentiment_Best_Model_2.pt")
                    
def test_Best_LSTM_Model(model_path, test_loader, criterion, train_on_gpu, vocab_size,
                         output_size, embedding_dim, hidden_dim, n_layers, batch_size):
    
    # Test the model on test dataset
    # Get test data loss and accuracy
    # Load best model
    LSTM_test = LSTM_Model(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    LSTM_test.load_state_dict(torch.load(model_path))
    LSTM_test.cuda()

    test_losses = [] # track loss
    num_correct = 0

    # init hidden state
    h = LSTM_test.init_hidden(batch_size)

    LSTM_test.eval()
    # iterate over test data
    with torch.no_grad():
        for inputs, labels in test_loader:

            # Creating new variables for the hidden state
            h = tuple([each.data for each in h])

            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
    
            # get predicted outputs
            output, h = LSTM_test(inputs.long(), h)
    
            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())
    
            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer
    
            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)
    return test_losses, num_correct