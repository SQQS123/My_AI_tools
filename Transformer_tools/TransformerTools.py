import math
import pandas as pd
import torch
from torch import nn
from AttentionTools import DotProductAttention,MultiHeadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)/torch.pow(10000, torch.arange(0, num_hiddens, 2,dtype=torch.float32)/num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.register_buffer('pe', self.P)

    def forward(self, X):
        x = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(x)

# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
    
# 残差连接和层规范化
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)

        # 这里用的是pytorch自带的注意力机制工具，我们用自己实现的来学习
        # self.attention = nn.MultiheadAttention(embed_dim=num_hiddens,
        #                                        num_heads=num_heads,
        #                                        dropout=dropout,
        #                                        bias=use_bias)

        self.attention = DotProductAttention(dropout)

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 多头自注意力机制的输出
        attn_output, _ = self.attention(X, X, X, valid_lens)
        # 第一个残差连接和层规范化
        Y = self.addnorm1(X, attn_output)
        # 基于位置的前馈网络的输出
        ffn_output = self.ffn(Y)
        # 第二个残差连接和层规范化
        return self.addnorm2(Y, ffn_output)
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers,
                 dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, dropout,
                             use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值范围在-1到1之间，所以嵌入值乘以嵌入维度的平方根进行缩放，再与位置编码相加
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i,blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, i,
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)

        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理
        # 因此state[2][self.i]最初为None
        # 预测阶段，输出序列是通过词元一个接一个地解码的
        # 因此state[2][self.i]包含着直到当前时间步的第i个块的所有输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size, 1, num_steps)
            # 它的每一行都是[1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # 自注意力机制
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力机制
        # enc_outputs的开头:(batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
    
class TransformerDecoder():
    pass


if __name__ == "__main__":
    pass
    # 下面是验证上面的代码功能的代码

    # ffn = PositionWiseFFN(ffn_num_input=4, ffn_num_hiddens=4, ffn_num_outputs=8)
    # ffn.eval()
    # print(ffn(torch.ones((2, 3, 4)))[0])
    # 完成

    # add_norm = AddNorm([3,4],0.5)
    # add_norm.eval()
    # print(add_norm(torch.ones((2,3,4)), torch.ones((2,3,4))).shape)
    # 完成

    X = torch.ones((2,100,24))
    valid_lens = torch.tensor([3,2])
    encoder_blk = EncoderBlock(24,24,24,24,[100,24],24,48,8,0.5)
    encoder_blk.eval()


    # encoder = TransformerEncoder(200,24,24,24,24,[100,24],24,48,8,2,0.5)
    # encoder.eval()
    # valid_lens = torch.tensor([3,2])
    # print(encoder(torch.ones((2,100),dtype=torch.long),valid_lens).shape)
    # 完成

    decoder_blk = DecoderBlock(24,24,24,24,[100,24],24,48,8,0.5,0)
    decoder_blk.eval()
    X = torch.ones((2,100,24))
    state = [encoder_blk(X,valid_lens), valid_lens, [None]]
    print(decoder_blk(X,state)[0].shape)