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
        
        >>> batch_size = 3
        >>> input_size = 10
        >>> hidden_size = 20
        >>> rnn = RNNCell(input_size, hidden_size)
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
        bias:
            If `False`, then the layer does not use bias.
        activation:
            The activation function to use. Can be either `tanh` or `relu`.
            
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
        self.bias = bias
        
        self.rnn_cell = RNNCell(input_size, hidden_size, bias, activation)
    
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
                A tensor containing the output features from the RNN layer.
                    
        Shape:
            inputs: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(H_{out})` or `(N, H_{out})`
            Returns: 
                output: `(L, H_{out})` or `(N, L, H_{out})`
            
            where
                N is a batch size.
                L is a sequnece length.
                H_{in} is a input size.
                H_{out} is a hidden size.
        """
        if inputs.dim() == 3:
            # Transpose inputs tensor for simpler compute code.
            # `(N, L, H_{in})` -> `(L, N, H_{in})`
            inputs = inputs.transpose(0, 1).contiguous()
        
        output = []
        h_i = hidden
        
        for x_i in inputs:
            h_i = self.rnn_cell(x_i, h_i)
            output.append(h_i)
        
        output = torch.stack(output, -2) # `(N, L, H_{out})`
        
        return output
    

class RNN(nn.Module):
    """
    This class is a implementation of multi-layer RNN.
    
    Args:
        input_size:
            The number of expected features in the input `x`.
        hidden_size:
            The number of features in the hidden state `h`.
        num_layers:
            The number of RNN layers. E.g. setting `num_layers=2`,
            it would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results.
        bias:
            If `False`, then the layer does not use bias.
        activation:
            The activation function to use. Can be either `tanh` or `relu`.
    
    Examples:
        
        >>> input_size = 10
        >>> hidden_size = 20
        >>> num_layers = 2
        >>> batch_size = 3
        >>> len_sequence = 5
        >>> rnn = RNN(input_size, hidden_size, num_layers)
        >>> inputs = torch.randn(batch_size, len_sequence, input_size)
        >>> hidden = torch.randn(batch_size, num_layers, hidden_size)
        >>> output = rnn(inputs, hidden)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        # Stacking RNN layers
        first_layer = RNNLayer(input_size, hidden_size, bias, activation)
        rnn_layer = RNNLayer(hidden_size, hidden_size, bias, activation)
        layers = [first_layer] + [rnn_layer for _ in range(1, num_layers)]
        self.layers = nn.ModuleList(layers)
        
    def forward(
        self,
        inputs: Tensor,
        hidden: Optional[Tensor] = None
    ) -> Tensor:
        """
        Pass inputs through a RNN.
        
        Args:
            inputs:
                A tensor containing input features.
            hidden:
                A tensor containing the initial hidden state per layer.
                
        Returns:
            output:
                A tensor containing the output features from the RNN.
                    
        Shape:
            inputs: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(num_layers, H_{out})` or `(N, num_layers, H_{out})`
            Returns: 
                output: `(L, H_{out})` or `(N, L, H_{out})`
            
            where
                N is a batch size.
                L is a sequnece length.
                H_{in} is a input size.
                H_{out} is a hidden size.
                num_layers is a number of layers.
        """
        if hidden.dim() == 3:
            # Transpose hidden tensor for simpler compute code.
            # `(N, num_layers, H_{out})` -> `(num_layers, N, H_{out})`
            hidden = hidden.transpose(0, 1).contiguous()
            
        output = inputs
        for i, rnn_layer in enumerate(self.layers):
            output = rnn_layer(output, hidden[i])
       
        return output