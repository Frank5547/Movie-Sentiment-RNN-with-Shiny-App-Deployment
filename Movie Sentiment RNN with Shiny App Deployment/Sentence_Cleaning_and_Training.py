# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:58:49 2019
Author: Francisco Javier Carrera Arias
LSTM_Model Data Preprocessing and Training Loop
"""
# Imports
import torch
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
# LSTM Model import
from LSTM_Model import LSTM_Model
from utils import pad_features, prune_review_text, train_LSTM_Net, test_Best_LSTM_Model
    
# Read Reviews
review_data = pd.read_csv("imdb_data.csv", encoding = "latin1")
reviews = list(review_data["review"].values)
labels = list(review_data["label"].values)
    
# Clean the review text
clean_reviews = prune_review_text(reviews)

# Gather all the words to build a vocabulary
all_words = [word for review in clean_reviews for word in review]

wordCounts = Counter(all_words)
sortCounts = sorted(wordCounts, key=wordCounts.get, reverse=True)

## Build a dictionary that maps words to integers
int_2_vocab = {k + 1:word for k, word in enumerate(sortCounts)}
vocab_2_int = {word: k for k, word in int_2_vocab.items()}

###################################################
# Save Vocabulary for later use
with open('vocabulary_to_int.pkl', 'wb') as f:
        pickle.dump(vocab_2_int, f)
###################################################
        
## use the vocab_2_int dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for k in range(len(clean_reviews)):
    reviews_ints.append([vocab_2_int[j] for j in clean_reviews[k]])

# Encode the labels
encoded_labels = np.array([1 if label == 'pos' else 0 for label in labels])
print(labels[:10])
print(encoded_labels[:10])

# Truncate and and pad very long reviews
seq_length = 200
review_features = pad_features(reviews_ints, seq_length=seq_length)

## test statements##
assert len(review_features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(review_features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 50 batches 
print(review_features[:50,:10])

# Split into training, validation and test sets
split_frac = 0.8
test_x, train_x, test_y, train_y = train_test_split(review_features, encoded_labels, test_size=split_frac, random_state=42)
test_x, val_x, test_y, val_y = train_test_split(test_x,test_y,test_size = 0.5,random_state = 42)
## print out the shapes of your resultant feature data
print(train_x.shape)
print(test_x.shape)
print(val_x.shape)

# Make DataLoaders
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Visualize a batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

# GPU flag
train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Instantiate the neural network with hyperparameters
vocab_size = len(vocab_2_int)+1
output_size = 1
embedding_dim = 250
hidden_dim = 512
n_layers = 2

LSTM = LSTM_Model(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(LSTM)

# Optimizer and Loss functions
lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(LSTM.parameters(), lr=lr)
epochs = 3
# Training Loop
train_LSTM_Net(LSTM, train_loader, valid_loader, batch_size, criterion, optimizer, epochs, train_on_gpu)

########################################################################################
# Test trained models

test_losses, num_correct = test_Best_LSTM_Model("Sentiment_Best_Model.pt", test_loader, criterion, train_on_gpu, vocab_size,
                                                output_size, embedding_dim, hidden_dim, n_layers, batch_size)

# Print test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))
# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))