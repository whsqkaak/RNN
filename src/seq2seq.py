import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Union, Optional, List, Iterable, Tuple

from rnn import RNN
from lstm import LSTM
from gru import GRU


class Encoder(nn.Module):
    """
    This class is implementation of Seq2Seq model encoder.
    The Encoder is composed of a stack of N identical RNN based Neural Network layers.
    
    Args:
        input_dim:
            The number of expected features in the inputs. The dimension of input features.
        emb_dim:
            The dimension of the embedding layer.
        hid_dim:
            The number of features in the hidden state. The dimension of hidden state.
        num_layers:
            The number of layers in the RNN.
        model_type:
            The type of RNN based Neural Network.
            It can be `RNN`, `LSTM` or `GRU`.
        dropout:
            The dropout value.
        bias:
            If `False`, then the rnn layer does not use bias.
        activation:
            The activation function to use in the rnn. Can be either `tanh` or `relu`.
    """
    
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hid_dim: int,
        num_layers: int = 2,
        model_type: str = "rnn",
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_Dim = hid_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        if model_type in ("rnn", "RNN"):
            self.rnn = RNN(emb_dim, hid_dim, num_layers, bias, activation)
        elif model_type in ("lstm", "LSTM"):
            self.rnn = LSTM(emb_dim, hid_dim, num_layers, bias, activation)
        elif model_type in ("gru", "GRU"):
            self.rnn = GRU(emb_dim, hid_dim, num_layers, bias, activation)
        else:
            raise RuntimeError(
                f"Unknown model type: {model_type}. It must be `rnn`, `lstm` or `gru`")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        source: Tensor,
        hidden: Optional[Tensor] = None,
        cell: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]
        """
        Pass source sequence through the encoder.
        
        Args:
            source:
                A input sequence. The iterable of tokens sequence.
            hidden:
                A initial hidden state. If None, it will be created in RNN.
            cell:
                A initial cell state.
        
        Return:
            h_n or (h_n, c_n)
            h_n:
                A tensor containing the last hidden state from the encoder RNN.
            c_n:
                A tensor containing the last cell state from the encoder RNN(Only LSTM).
                
        Shape:
            source: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(H_{out})` or `(N, H_{out})`
            cell: `(H_out})` or `(N, H_{out})`
            Return:
                h_n: `(H_{out})` or `(N, H_{out})`
                c_n: `(H_{out})` or `(N, H_{out})`
                
            where
                N is a batch size.
                L is a sequence length.
                H_{in} is a dimension of input.
                H_{out} is a dimension of hidden.
        """
        is_batched = source.dim() == 3
        
        if not is_batched:
            # `(L, H_{in})` -> `(1, L, H_{in})`
            source = source.unsqueeze(0)
            
        # `(N, L, H_{in})` -> `(N, L, H_{emb})`
        # where H_{emb} is a dimension of embedding.
        emb_source = self.embedding(source)
        emb_source = self.dropout(emb_source)
        
        if self.model_type in ("lstm", "LSTM"):
            output, (h_n, c_n) = self.rnn(emb_source, hidden, cell)
            return h_n, c_n
        else:
            output = self.rnn(emb_source, hidden)
            if is_batched:
                # `(N, L, H_{out})` -> `(L, N, H_{out})`
                output = output.transpose(0, 1).contiguous()
            h_n = output[-1]
            return h_n
        
        
class Decoder(nn.Module):
    # TODO: Decoder에 맞게 구조 변경
    """
    This class is implementation of Seq2Seq model decoder.
    The Decoder is composed of a stack of N identical RNN based Neural Network layers.
    
    Args:
        input_dim:
            The number of expected features in the inputs. The dimension of input features.
            In Decoder, it will be dimension of output features.
        emb_dim:
            The dimension of the embedding layer.
        hid_dim:
            The number of features in the hidden state. The dimension of hidden state.
        num_layers:
            The number of layers in the RNN.
        model_type:
            The type of RNN based Neural Network.
            It can be `RNN`, `LSTM` or `GRU`.
        dropout:
            The dropout value.
        bias:
            If `False`, then the rnn layer does not use bias.
        activation:
            The activation function to use in the rnn. Can be either `tanh` or `relu`.
    """
    
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hid_dim: int,
        num_layers: int = 2,
        model_type: str = "rnn",
        dropout: float = 0.1,
        bias: bool = True,
        activation: str = "tanh"
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_Dim = hid_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        if model_type in ("rnn", "RNN"):
            self.rnn = RNN(emb_dim, hid_dim, num_layers, bias, activation)
        elif model_type in ("lstm", "LSTM"):
            self.rnn = LSTM(emb_dim, hid_dim, num_layers, bias, activation)
        elif model_type in ("gru", "GRU"):
            self.rnn = GRU(emb_dim, hid_dim, num_layers, bias, activation)
        else:
            raise RuntimeError(
                f"Unknown model type: {model_type}. It must be `rnn`, `lstm` or `gru`")
        
        self.output_layer = nn.Linear(hid_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        target: Tensor,
        hidden: Tensor,
        cell: Optional[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        """
        Pass target sequence through the decoder.
        
        Args:
            target:
                A input sequence. The iterable of tokens sequence.
            hidden:
                A hidden state from the encoder.
            cell:
                A cell state from the encoder.
        
        Return:
            (output, h_n) or (output, (h_n, c_n))
            output:
                A tensor containing the output features from the decoder RNN.
            h_n:
                A tensor containing the last hidden state from the decoder RNN.
            c_n:
                A tensor containing the last cell state from the decoder RNN(Only LSTM).
                
        Shape:
            target: `(L, H_{in})` or `(N, L, H_{in})`
            hidden: `(H_{out})` or `(N, H_{out})`
            cell: `(H_out})` or `(N, H_{out})`
            Return:
                output: `(L, H_{in})` or `(N, L, H_{in})`
                h_n: `(H_{out})` or `(N, H_{out})`
                c_n: `(H_{out})` or `(N, H_{out})`
                
            where
                N is a batch size.
                L is a sequence length.
                H_{in} is a dimension of input.
                H_{out} is a dimension of hidden.
        """
        is_batched = target.dim() == 3
        
        if not is_batched:
            # `(L, H_{in})` -> `(1, L, H_{in})`
            target = target.unsqueeze(0)
            
        # `(N, L, H_{in})` -> `(N, L, H_{emb})`
        # where H_{emb} is a dimension of embedding.
        emb_target = self.embedding(target)
        emb_target = self.dropout(emb_target)
        
        if self.model_type in ("lstm", "LSTM"):
            # `(N, L, H_{emb})` -> `(N, L, H_{out})`
            output, (h_n, c_n) = self.rnn(emb_target, hidden, cell)
            
            # `(N, L, H_{out})` -> `(N, L, H_{in})`
            output = self.output_layer(output)
            return output, (h_n, c_n)
        else:
            # `(N, L, H_{emb})` -> `(N, L, H_{out})`
            output = self.rnn(emb_target, hidden)
            if is_batched:
                # `(N, L, H_{out})` -> `(L, N, H_{out})`
                output = output.transpose(0, 1).contiguous()
            h_n = output[-1]
            
            # `(N, L, H_{out})` -> `(N, L, H_{in})`
            output = self.output_layer(output)
            return output, h_n
        

class Seq2Seq(nn.Module):
    """
    This class is implementation of Seq2Seq model.
    
    Args:
        encoder:
            The encoder of Seq2Seq model.
        decoder:
            The decoder of Seq2Seq model.
    """
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.model_type = encoder.model_type
        
        assert encoder.hid_dim == decoder.hid_dim, \
            f"Hidden dimensions of encoder: {encoder.hid_dim} and decoder: {decoder.hid_dim} must be equall."
        assert encoder.num_layers == decoder.num_layers, \
            f"""
            Encoder and Decoder must have equal number of layers!
            {encoder.num_layers} != {decoder.num_layers}
            """
        assert encoder.model_type == decoder.model_type, \
            f"Encoder and Decoder must have same model type. {encoder.model_type} != {decoder.model_type}"
        
        
    def forward(
        self,
        source: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        Pass soruce sequence and target sequence through the Seq2Seq model.
        
        Args:
            source:
                The source sequence.
            target:
                The target sequence.
                
        Return:
            A tensor containing the predictions.
            
        Shape:
            source: `(L_s, H_{in})` or `(N, L_s, H_{in})`
            target: `(L_t, H_{out})` or `(N, L_t, H_{out})`
            Return:
                `(L_t, H_{out})` or `(N, L_t, H_{out})`
            
            where
                N is a batch size.
                L_s is a source sequence length.
                L_t is a target sequence length.
                H_{in} is a dimension of input.
                H_{out} is a dimension of hidden.
        """
        is_batched = source.dim() == 3
        
        if not is_batched:
            source = source.unsqueeze(0)
            target = source.unsqueeze(0)
            
        batch_size, target_len, output_size = target.shape
        
        if self.model_type in ("lstm", "LSTM"):
            hidden, cell = self.encoder(source)
            outputs, hidden, cell = self.decoder(target, hidden, cell)
        
        else:
            hidden = self.encoder(source)
            outputs, hidden = self.decoder(target, hidden)
            
        return outputs