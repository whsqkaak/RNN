import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
from typing import Optional, Tuple


class LSTMCell(nn.Module):
    """
    This class is implementation of LSTM Cell.
    
    Args:
        input_size:
            The number of expected features in the input `x`
        hidden_size:
            The number of features in the hidden state `h`
        bias:
            If `False`, then the layer does not use bias.
            
    Examples:
        
        >>> batch_size = 3
        >>> input_size = 10
        >>> hidden_size = 20
        >>> lstm = LSTMCell(input_size, hidden_size)
        >>> inputs = torch.randn(batch_size, input_size)
        >>> h_0 = torch.randn(batch_size, hidden_size)
        >>> c_0 = torch.randn(batch_size, hidden_size)
        >>> h_1, c_1 = lstm(inputs, (h_0, c_0))
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
    def forward(
        self,
        inputs: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Pass inputs through a LSTM Cell.
        
        Args:
            inputs:
                A tensor containing input features. `x`
            state:
                A tuple containing the initial hidden state and the initial cell state. `(h_0, c_0)`
                
        Returns:
            A tuple containing the next hidden state and the next cell state 
            for each element in the batch. `(h_1, c_1)`
                    
        Shape:
            inputs: `(H_{in})` or `(N, H_{in})`
            state: `(h_0, c_0)`
                h_0: `(H_{out})` or `(N, H_{out})`
                c_0: `(H_{out})` or `(N, H_{out})`
            Returns: `(h_1, c_1)`
                h_1: `(H_{out})` or `(N, H_{out})`
                c_1: `(H_{out})` or `(N, H_{out})`
            
            where
                N is a batch size.
                H_{in} is a input size.
                H_{out} is a hidden size.
        """
        assert inputs.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {inputs.dim()}-D tensor"
        is_batched = inputs.dim() == 2
        if not is_batched:
            # `(H_{in})` -> `(1, H_{in})`
            inputs = inputs.unsqueeze(0)
        
        if state is None:
            # Create Initial state
            # `(N, H_{out})`
            zeros = torch.zeros(inputs.size(0), self.hidden_size, dtype=inputs.dtype)
            state = (zeros, zeros)
        else:
            # If not batched,
            # `(H_{out})` -> `(1, H_{out})`
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0)) if not is_batched else state
            
        h_0, c_0 = state
        
        gate_in = torch.sigmoid(self.create_gate(inputs, h_0))
        gate_forget = torch.sigmoid(self.create_gate(inputs, h_0))
        gate_cell = torch.tanh(self.create_gate(inputs, h_0))
        gate_out = torch.sigmoid(self.create_gate(inputs, h_0))
        
        next_cell_state = (gate_forget * c_0) + (gate_in * gate_cell)
        next_hidden_state = gate_out * torch.tanh(next_cell_state)
        
        if not is_batched:
            # `(1, H_{out})` -> `(H_{out})`
            next_hidden_state = next_hidden_state.squeeze(0)
            next_cell_state = next_cell_state.squeeze(0)
            
        return (next_hidden_state, next_cell_state)
        
    def create_gate(
        self,
        inputs: Tensor,
        hidden: Tensor
    ) -> Tensor:
        linear_input = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        linear_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        return linear_input(inputs) + linear_hidden(hidden)