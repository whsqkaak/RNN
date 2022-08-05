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
            hidden: `(H_{out})` or `(N, H_{out})`
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
    
    
class GRULayer(nn.Module):
    """
    This class is implementation of GRU Layer.
    
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
        >>> len_sequence = 8
        >>> gru = GRULayer(input_size, hidden_size)
        >>> inputs = torch.randn(batch_size, len_sequence, input_size)
        >>> h_0 = torch.randn(batch_size, hidden_size)
        >>> output = gru(inputs, h_0)
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
        self.cell = GRUCell(
            input_size,
            hidden_size,
            bias
        )
        
    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor
    ) -> Tensor:
        """
        Pass inputs through a GRU Layer.
        
        Args:
            inputs:
                A tensor containing input features. `x`
            hidden:
                A tensor contatining the initial hidden state. `h_0`
            
        Returns:
            A tensor containing output features from the GRU Layer.
        
        Shape:
            inputs: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(H_{out})` or `(N, H_{out})`
            Returns:
                output: `(L, H_{out})` or `(N, L, H_{out})`
            
            where
                N is a batch size.
                L is a sequence length.
                H_{in} is a input size.
                H_{out} is a hidden size.
        """
        if inputs.dim() == 3:
            # Transpose inputs tensor for simpler compute code.
            # `(N, L, H_{in})` -> `(L, N, H_{in})`
            inputs = inputs.transpose(0, 1).contiguous()
            
        output = []
        
        for x_i in inputs:
            hidden = self.cell(x_i, hidden)
            output.append(hidden)
            
        output = torch.stack(output, -2) # `(N, L, H_{out})`
        return output
    
    
class GRU(nn.Module):
    """
    This class is implementation of GRU.
    
    Args:
        input_size:
            The number of expected features in the input `x`
        hidden_size:
            The number of featrues in the hidden state `h`
        num_layers:
            The number of GRU layers.
        bias:
            If `False`, the the layer does not use bias.
            
    Examples:
        
        >>> batch_size = 3
        >>> input_size = 10
        >>> hidden_size = 20
        >>> len_sequence = 8
        >>> num_layers = 2
        >>> gru = GRULayer(input_size, hidden_size, num_layers)
        >>> inputs = torch.randn(batch_size, len_sequence, input_size)
        >>> h_0 = torch.randn(batch_size, num_layers, hidden_size)
        >>> output = gru(inputs, h_0)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacking GRU layers.
        first_layer = GRULayer(input_size, hidden_size, bias)
        gru_layer = GRULayer(hidden_size, hidden_size, bias)
        layers = [first_layer] + [gru_layer for _ in range(1, num_layers)]
        self.layers = nn.ModuleList(layers)
        
    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor
    ) -> Tensor:
        """
        Pass inputs through a GRU.
        
        Args:
            inputs:
                A tensor containing input features. `x`
            hidden:
                A tensor contatining the initial hidden state. `h_0`
            
        Returns:
            A tensor containing output features from the GRU.
        
        Shape:
            inputs: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(num_layers, H_{out})` or `(N, num_layers, H_{out})`
            Returns:
                output: `(L, H_{out})` or `(N, L, H_{out})`
            
            where
                N is a batch size.
                L is a sequence length.
                H_{in} is a input size.
                H_{out} is a hidden size.
                num_layers is a number of layers.
        """
        if hidden.dim() == 3:
            # Transpose hidden tensor for simpler compute code.
            # `(N, L, H_{out})` -> `(L, N, H_{out})`
            hidden = hidden.transpose(0, 1).contiguous()
            
        output = inputs
        for i, layer in enumerate(self.layers):
            output = layer(output, hidden[i])
            
        return output