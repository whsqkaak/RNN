"""
Usage:
    cd RNN; pytest test/test_RNN.py
"""

import pytest
import torch

from src.rnn import (
    RNNCell,
    RNNLayer
)

def rnn_cell_test():
    # Prepare RNNCell instance and dummy data.
    batch_size = 3
    input_size = 10
    hidden_size = 20
    
    inputs = torch.randn(batch_size, input_size)
    hidden = torch.randn(batch_size, hidden_size)
    
    rnn_cell = RNNCell(input_size, hidden_size, bias=True, activation="tanh")
    
    output = rnn_cell(inputs, hidden)
    
    # The shape of output must be `(batch_size, hidden_size)`
    assert output.shape == (batch_size, hidden_size)
    
    
def rnn_layer_test():
    # Prepare RNN instance and dummy data.
    batch_size = 3
    input_size = 10
    hidden_size = 20
    len_sequence = 8
    
    inputs = torch.randn(batch_size, len_sequence, input_size)
    hidden = torch.randn(batch_size, hidden_size)
    
    rnn_layer = RNNLayer(input_size, 
              hidden_size,
              bias=True,
              activation="tanh"
             )
    
    output = rnn_layer(inputs, hidden)
    
    # Check the shape of output and h_n
    assert output.shape == (batch_size, len_sequence, hidden_size)


def test_main():
    rnn_cell_test()
    rnn_layer_test()