# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 18:32:52 2019
Author: Francisco Javier Carrera Arias
predict_new_review
"""
# Imports
import torch
import pickle
import numpy as np
# Custom imports
from LSTM_Model import LSTM_Model
from utils import pad_features, prune_review_text

def predict_new_review(review, vocabulary_path, best_model_path):

    pruned_review = prune_review_text(review, single_rev = True, download_stopwords = True)

    # Import Vocabulary
    with open(vocabulary_path, 'rb') as f:
        vocab_2_int = pickle.load(f)

    # Delete words not in original vocabulary
    clean_review = [[word for word in pruned_review if word in list(vocab_2_int.keys())]]
    ## use the vocab_2_int dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints
    reviews_ints = []
    for k in xrange(len(clean_review)):
        reviews_ints.append([vocab_2_int[j] for j in clean_review[k]])
    # Pad Review
    padded_review = pad_features(reviews_ints, 200)
    numpy_review = np.array(padded_review)

    # Instantiate the best LSTM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab_2_int)+1
    output_size = 1
    embedding_dim = 250
    hidden_dim = 512
    n_layers = 2

    LSTM = LSTM_Model(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    LSTM.load_state_dict(torch.load(best_model_path, map_location = device))

    # Predict the sentiment of the new review using the LSTM model
    data = torch.from_numpy(numpy_review)
    data = data.to(device)

    h = LSTM.init_hidden(1)
    LSTM.to(device).eval()

    with torch.no_grad():
        h = tuple([each.data for each in h])
        out, h = LSTM(data.long(), h)
    
    result = np.round(out.cpu().numpy())
    
    return result
