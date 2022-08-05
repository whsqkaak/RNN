"""
Usage:
    cd RNN; pytest test/test_GRU.py
"""

import pytest
import torch

from src.gru import (
    GRUCell,
    GRULayer,
    GRU
)

def gru_cell_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    
    inputs = torch.randn(batch_size, input_size)
    hidden = torch.randn(batch_size, hidden_size)
    
    gru = GRUCell(input_size, hidden_size)
    
    next_hidden = gru(inputs, hidden)
    
    assert next_hidden.shape == (batch_size, hidden_size)

def gru_layer_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    len_sequence = 8
    
    inputs = torch.randn(batch_size, len_sequence, input_size)
    hidden = torch.randn(batch_size, hidden_size)
    
    gru = GRULayer(input_size, hidden_size)
    
    output = gru(inputs, hidden)
    
    assert output.shape == (batch_size, len_sequence, hidden_size)
    
def gru_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    len_sequence = 8
    num_layers = 2
    
    inputs = torch.randn(batch_size, len_sequence, input_size)
    hidden = torch.randn(batch_size, num_layers, hidden_size)
    
    gru = GRU(input_size, hidden_size, num_layers)
    
    output = gru(inputs, hidden)
    
    assert output.shape == (batch_size, len_sequence, hidden_size)
    
def test_main():
    # gru_cell_test()
    # gru_layer_test()
    gru_test()