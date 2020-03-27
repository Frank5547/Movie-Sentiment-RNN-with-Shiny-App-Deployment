# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:50:40 2019
Author: Francisco Javier Carrera Arias
LSTM_Model
"""
import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.4):
        """
        Initialize the model by setting up the layers.
        """
        super(LSTM_Model, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # Embedding Layer
        self.Embed = nn.Embedding(vocab_size, embedding_dim)
        # LSTM Layers
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = drop_prob, batch_first = True)
        # Fully Connected Layers
        self.FC1 = nn.Linear(hidden_dim, output_size)
        # Sigmoid Activation Function
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform forward pass using input and hidden states.
        """
        batch_size = x.size(0)
        x = self.Embed(x)
        x, hidden = self.LSTM(x,hidden)
        x = x.contiguous().view(-1, self.hidden_dim)
        sig_out = self.sig(self.FC1(x))
        
        # batch size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden