"""
Usage:
    cd RNN; pytest test/test_GRU.py
"""

import pytest
import torch

from src.gru import (
    GRUCell,
    GRULayer,
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
    
    gru = GRULayer(input_size, hidden)
    
    h_n = gru(inputs, hidden)
    
    assert h_n.shape == (batch_size, len_sequence, hidden_size)
    
def test_main():
    gru_cell_test()