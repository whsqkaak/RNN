import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import copy
from typing import Optional


class RNNCell(nn.Module):
    """
    This class is a implementation of Simple RNN cell.
    
    Args:
        input_size:
            The number of expected features in the input `x`.
        hidden_size:
            The number of features in the hidden state `h`.
        bias:
            If `False`, then the layer does not use bias.
        activation:
            The activation function to use. Can be either `tanh` or `relu`.
            
    Examples:
        
        >>> rnn = RNNCell(10, 20)
        >>> batch_size = 3
        >>> input_size = 10
        >>> hidden_size = 20
        >>> inputs = torch.randn(batch_size, input_size)
        >>> hidden = torch.randn(batch_size, hidden_size)
        >>> hx = rnn(inputs, hidden)
        
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_linear = nn.Linear(input_size, hidden_size, bias=bias) # `W_{ih} * x + b_{ih}`
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=bias) # `W_{hh} * h + b_{hh}`
        
        # Setting activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise RuntimeError(
                f"Unknown activation function: {activation}. It must be `tanh` or `relu`")
                
    def forward(
        self,
        inputs: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass inputs through a RNN Cell.
        
        Args:
            inputs:
                A tensor containing input features.
            hidden:
                A tensor containing the initial hidden state.
                
        Returns:
            A tensor containing the next hidden state for each element in the batch.
                    
        Shape:
            inputs: `(H_{in})` or `(N, H_{in})`
            hidden: `(H_{out})` or `(N, H_{out})`
            Returns: `(H_{out})` or `(N, H_{out})`
            
            where
                N is a batch size.
                H_{in} is a input size.
                H_{out} is a hidden size.
        """
        dim_inputs = inputs.dim()
        assert dim_inputs in (1, 2), \
            f"RNNCell inputs to be 1-D or 2-D but received {dim_inputs}-D tensor"
        is_batched = dim_inputs == 2
        if not is_batched:
            # `(H_{in})` -> `(1, H_{in})`
            inputs = inputs.unsqueeze(0)
            
        if hidden is None:
            # Create initial hidden state.
            # `(N, H_{out})`
            hidden = torch.zeros(inputs.shape[0], self.hidden_size, dtype=inputs.dtype)
        else:
            # If not batched, `(H_{out})` -> `(1, H_{out})`
            hidden = hidden.unsqueeze(0) if not is_batched else hidden
        
        input_gate = self.input_linear(inputs) # `(N, H_{int})` -> `(N, H_{out})`
        hidden_gate = self.hidden_linear(hidden) # `(N, H_{out})` -> `(N, H_{out})`, No dimension change.
        output = self.activation(input_gate + hidden_gate)
        
        if not is_batched:
            # `(1, H_{out})` -> `(H_{out})`
            output = output.squeeze(0)
        
        return output
    

class RNNLayer(nn.Module):
    """
    This class is a implementation of RNN Layer.
    
    Args:
        input_size:
            The number of expected features in the input `x`.
        hidden_size:
            The number of features in the hidden state `h`.
        num_classes:
            The number of output classes.
        bias:
            If `False`, then the layer does not use bias.
        activation:
            The activation function to use. Can be either `tanh` or `relu`.
            
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        bias: bool = True,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Stacking RNN layers
        self.rnn_cell = RNNCell(input_size, hidden_size, bias, activation)
        
        # Setting output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(
        self,
        inputs: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass inputs through a RNN Layer.
        
        Args:
            inputs:
                A tensor containing input features.
            hidden:
                A tensor containing the initial hidden state.
                
        Returns:
            output:
                A tensor containing the output features from the last layer of the RNN.
            h_n:
                A tensor containing the final hidden state for each element in the batch.
                    
        Shape:
            inputs: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(L, H_{out})` or `(N, L, H_{out})`
            Returns: 
                output: `(L, num_classes)` or `(N, L, num_classes)`
                h_n: `(L, H_{out})` or `(N, L, H_{out})`
            
            where
                N is a batch size.
                L is a sequnece length.
                H_{in} is a input size.
                H_{out} is a hidden size.
                num_classes is a number of classes.
        """
        is_batched = inputs.dim() == 3
        if not is_batched:
            # `(L, H_{in})` -> `(1, L, H_{in})`
            inputs = inputs.unsqueeze(0)
        len_sequence = inputs.shape[1]
            
        if hidden is None:
            # Create initial hidden state.
            # `(N, L, H_{out})`
            hidden = torch.zeros(inputs.shape[0], len_sequence, self.hidden_size, dtype=inputs.dtype)
        else:
            # If not batched, `(L, H_{out})` -> `(1, L, H_{out})`
            hidden = hidden.unsqueeze(0) if not is_batched else hidden
            
        h_n = []
        h_i = hidden[:, 0, :]
        
        for i in range(len_sequence):
            h_i = self.rnn_cell(inputs[:,i,:], h_i)
            h_n.append(h_i)
        
        h_n = torch.stack(h_n, 1) # `(N, L, H_{out})`
        
        output = self.output_layer(h_n)
        
        if not is_batched:
            # `(1, L, num_classes)` -> `(L, num_classes)`
            output = output.squeeze(0)
            
            # `(1, L, H_{out})` -> `(L, H_{out})`
            h_n = h_n.squeeze(0)
        
        return output, h_n