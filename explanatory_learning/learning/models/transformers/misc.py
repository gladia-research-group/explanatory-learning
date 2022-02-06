import math
import torch

class PositionalEncoding(torch.nn.Module):
    """Applies positional encoding of a sequence of input vectors.
    
    A sequence of positional encoding vectors with the same length and depth 
    as the input vectors is summed to the input sequence.
    For more information about how the positional encoding works see the 
    article from Vaswani et al (https://arxiv.org/pdf/1706.03762.pdf).

    :param d_model: depth of the input vectors
    :param dropout: dropout probability (applied after the positional encoding)
    :param maxlen: maximum possible length for the input sequence.

    """
    def __init__(self, d_model:int, dropout:float=0.1, max_len:int=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of shape :math:`(N, *, H)` with :math:`H` equals to the model depth.

        :return: tensor of shape :math:`(N, *, H)`  with :math:`H` equals to the model depth.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VisualizerDecoderLayer(torch.nn.TransformerDecoderLayer):
    """Custom decoder layer which stores attention values.

    This module is a variant of the *torch.nn.TransformerDecoderLayer* that
    stores the temporary attention matrices (averaged over all heads) for both 
    self-attention and attention with the encoder output. These can be used for 
    visualization purposes.

    .. note::

        See documentation for *torch.nn.TransformerDecoderLayer* for more information
        about this class.
    """
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int=2048, dropout:int=0.1, activation:str="relu"):
        super(VisualizerDecoderLayer,self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        self.cache_self_attn = None
        self.cache_mha_attn = None

    def forward(self,
            tgt,
            memory,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None):
        tgt2, self.cache_self_attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, self.cache_src_attn  = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)


        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    @property
    def self_attention(self):
        """(Averaged) self Attention of the decoder hidden state.
        """
        return self.cache_self_attn

    @property
    def multihead_attention(self):
        """(Averaged) multi-head Attention between encoder output and decoder hidden state.
        """
        return self.cache_mha_attn


class VisualizerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """Custom encoder layer which stores attention values.

    This module is a variant of the *torch.nn.TransformerEncoderLayer* that
    stores the temporary self attention matrix (averaged over all heads) of the layer. 
    This can be used for visualization purposes.

    .. note::

        Refer to the documentation for *torch.nn.TransformerEncoderLayer* for more information
        about this class.
    """
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int=2048, dropout:float=0.1, activation:str="relu"):
        super(VisualizerEncoderLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.cache_self_attn = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.cache_self_attn = self.self_attn(
            src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    @property
    def self_attention(self):
        """(Averaged) self Attention of the encoder layer hidden state.
        """
        return self.cache_self_attn