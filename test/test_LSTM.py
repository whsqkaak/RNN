"""
Usage:
    cd RNN; pytest test/test_LSTM.py
"""

import pytest
import torch

from src.lstm import (
    LSTMCell,
    LSTMLayer,
    LSTM
)

def lstm_cell_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    
    inputs = torch.randn(batch_size, input_size)
    h_0 = torch.randn(batch_size, hidden_size)
    c_0 = torch.randn(batch_size, hidden_size)
    
    lstm = LSTMCell(input_size, hidden_size)
    
    h_1, c_1 = lstm(inputs, (h_0, c_0))
    
    assert h_1.shape == (batch_size, hidden_size)
    assert c_1.shape == (batch_size, hidden_size)

def lstm_layer_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    len_sequence = 8
    
    inputs = torch.randn(batch_size, len_sequence, input_size)
    h_0 = torch.randn(batch_size, hidden_size)
    c_0 = torch.randn(batch_size, hidden_size)
    
    lstm = LSTMLayer(input_size, hidden_size)
    
    output, state = lstm(inputs, (h_0, c_0))
    h_n, c_n = state
    
    assert output.shape == (batch_size, len_sequence, hidden_size)
    assert h_n.shape == (batch_size, hidden_size)
    assert c_n.shape == (batch_size, hidden_size)

def lstm_test():
    batch_size = 3
    input_size = 10
    hidden_size = 20
    len_sequence = 8
    num_layers = 2
    
    inputs = torch.randn(batch_size, len_sequence, input_size)
    h_0 = torch.randn(batch_size, num_layers, hidden_size)
    c_0 = torch.randn(batch_size, num_layers, hidden_size)
    
    lstm = LSTM(input_size, hidden_size, num_layers)
    
    output, state = lstm(inputs, (h_0, c_0))
    h_n, c_n = state
    
    assert output.shape == (batch_size, len_sequence, hidden_size)
    assert h_n.shape == (batch_size, hidden_size)
    assert c_n.shape == (batch_size, hidden_size)
    
    
    
def test_main():
    lstm_cell_test()
    lstm_layer_test()
    lstm_test()