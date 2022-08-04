"""
Usage:
    cd RNN; pytest test/test_LSTM.py
"""

import pytest
import torch

from src.lstm import (
    LSTMCell
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

def test_main():
    lstm_cell_test()