import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
from typing import Optional, Tuple


class GRUCell(nn.Module):
    """
    This class is implementation of GRU Cell.
    
    Args:
        input_size:
            The number of expected features in the input `x`
        hidden_size:
            The number of featrues in the hidden state `h`
        bias:
            If `False`, the the layer does not use bias.
            
    Examples:
        
        >>> batch_size = 3
        >>> input_size = 10
        >>> hidden_size = 20
        >>> gru = GRUCell(input_size, hidden_size)
        >>> inputs = torch.randn(batch_size, input_size)
        >>> h_0 = torch.randn(batch_size, hidden_size)
        >>> h_1 = gru(inputs, h_0)
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
        hidden: Tensor
    ) -> Tensor:
        """
        Pass inputs through a GRU cell.
        
        Args:
            inputs:
                A tensor containing input features. `x`
            hidden:
                A tensor contatining the initial hidden state. `h_0`
            
        Returns:
            A tensor containing the next hidden state.
        
        Shape:
            inputs: `(H_{in})` or `(N, H_{in})`
            state: `(h_0, c_0)`
                h_0: `(H_{out})` or `(N, H_{out})`
            Returns:
                h_1: `(H_{out})` or `(N, H_{out})`
            
            where
                N is a batch size.
                H_{in} is a input size.
                H_{out} is a hidden size.
        """
        assert inputs.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {inputs.dim()}-D tensor"
        is_batched = inputs.dim() == 2
        if not is_batched:
            # `(H_{in})` -> `(1, H_{in})`
            inputs = inputs.unsqueeze(0)
        
        if hidden is None:
            # Create initial hidden state
            # `(N, H_{out})`
            hidden = torch.zeros(inputs.shape[0], self.hidden_size, dtype=inputs.dtype)
        else:
            # If not batched,
            # `(H_{out})` -> `(1, H_{out})`
            hidden = hidden.unsqueeze(0) if not is_batched else hidden
            
        gate_reset = torch.sigmoid(self.create_gate(inputs, hidden))
        gate_update = torch.sigmoid(self.create_gate(inputs, hidden))
        gate_new = torch.tanh(self.create_gate(inputs, hidden * gate_reset))
        
        next_hidden_state = gate_new + gate_update * (hidden - gate_new)
        
        if not is_batched:
            # `(1, H_{out})` -> `(H_{out})`
            next_hidden_state = next_hidden_state.squeeze(0)
        
        return next_hidden_state
        
    def create_gate(
        self,
        inputs: Tensor,
        hidden: Tensor
    ) -> Tensor:
        linear_input = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        linear_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        return linear_input(inputs) + linear_hidden(hidden)