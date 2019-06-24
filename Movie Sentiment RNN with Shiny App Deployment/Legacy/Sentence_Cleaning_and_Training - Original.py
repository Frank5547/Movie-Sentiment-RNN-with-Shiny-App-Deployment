# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:58:49 2019
Author: Francisco Javier Carrera Arias
LSTM_Model Data Preprocessing and Training Loop
"""
# Imports
import re
import torch
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
# LSTM Model import
from LSTM_Model import LSTM_Model

# Functions
def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    ## implement function
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
    
# Read Reviews
review_data = pd.read_csv("imdb_data.csv", encoding = "latin1")
reviews = list(review_data["review"].values)
labels = list(review_data["label"].values)
    
# Clean the review text
# Gather stopwords and Initializer stemmer
# nltk.download('stopwords') - download stopwords if needed
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
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

# Gather all the words to build a vocabulary
all_words = [word for review in clean_reviews for word in review]

wordCounts = Counter(all_words)
sortCounts = sorted(wordCounts, key=wordCounts.get, reverse=True)

## Build a dictionary that maps words to integers
int_2_vocab = {k + 1:word for k, word in enumerate(sortCounts)}
vocab_2_int = {word: k for k, word in int_2_vocab.items()}

# Save Vocabulary for later use
with open('vocabulary_to_int.pkl', 'wb') as f:
        pickle.dump(vocab_2_int, f)

## use the vocab_2_int dict to tokenize each review in reviews_split
## store the tokenized reviews in reviews_ints
reviews_ints = []
for k in range(len(clean_reviews)):
    reviews_ints.append([vocab_2_int[j] for j in clean_reviews[k]])

# Encode the labels
encoded_labels = np.array([1 if label == 'pos' else 0 for label in labels])
print(labels[15140:15145])
print(encoded_labels[15140:15145])

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

# Training Loop
epochs = 3
counter = 0
print_every = 100
clip = 5 # gradient clipping
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
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(LSTM.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = LSTM.init_hidden(batch_size)
            val_loss = 0
            LSTM.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = LSTM(inputs.long(), val_h)
                    vloss = criterion(output.squeeze(), labels.float())
                    val_loss += vloss.item()
            LSTM.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(val_loss/len(valid_loader)))
            valLossTracker.append(val_loss/len(valid_loader))
            if (val_loss/len(valid_loader)) == min(valLossTracker):
                print("Loss Decreased...Saving Checkpoint")
                torch.save(LSTM.state_dict(), "Sentiment_Best_Model.pt")

########################################################################################
# Load and test the best model
LSTM_test = LSTM_Model(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
LSTM_test.load_state_dict(torch.load("Sentiment_Best_Model.pt"))
LSTM_test.cuda()

# Test the model on test dataset
# Get test data loss and accuracy

test_losses = [] # track loss
num_correct = 0

# init hidden state
h = LSTM_test.init_hidden(batch_size)

LSTM_test.eval()
# iterate over test data
with torch.no_grad():
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
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

print("Test loss: {:.3f}".format(np.mean(test_losses)))
# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))