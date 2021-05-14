import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class AddPositionEmbs(nn.Module):
    """向输入中添加可学习的位置嵌入模块 """
    def __init__(self,inputs_positions=None):
        super(AddPositionEmbs, self).__init__()
        """
        默认情况下，这一层使用固定的sinusoidal embedding table。
        如果需要一个可学习的位置嵌入，将初始化器传递给posemb_init。
        Args:
            inputs: input data.
            inputs_positions: input position indices for packed sequences.
            posemb_init: positional embedding initializer.
        Returns:
            output: `(bs, timesteps, in_dim)`
        """
        self.inputs_positions = inputs_positions

    def forward(self, inputs):
        # inputs.shape is (batch_size, seq_len/embeddings num, emb_dim).
        assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                                  ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])#torch.Size([1, 65, 1024])
        pe = nn.Parameter(torch.randn(pos_emb_shape))#随机生成张量，尺寸为：torch.Size([1, 65, 1024])
        #print(pe)
        if self.inputs_positions is None:
            # Normal unpacked case:
            return inputs + pe
        else:
            # For packed data we need to use known position indices:
            return inputs + nn.take(pe[0], self.inputs_positions, axis=0)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block. 模块"""

    """
    Linear -> Gelu -> Dropout -> linear -> Dropout
    """
    def __init__(self, mlp_dim, Dim=1024, out_dim=None, dropout_rate=0.1):
        super(MlpBlock,self).__init__()
        self.out_dim = out_dim
        self.mlp_dim = mlp_dim
        self.Dim = Dim
        self.dropout_rate = dropout_rate

    def forward(self, inputs):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        output = nn.Sequential(
            nn.Linear(self.Dim, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_dim, actual_out_dim),
            nn.Dropout(self.dropout_rate)
        )
        return output(inputs)


class Attention(nn.Module):
    # attention
    def __init__(self, Dim = 1024, heads = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads  # 计算最终进行全连接操作时输入神经元的个数，64*16=1024
        project_out = not (heads == 1 and dim_head == Dim)  # 多头注意力并且输入和输出维度相同时为True

        self.heads = heads  # 多头注意力中“头”的个数
        self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍

        self.attend = nn.Softmax(dim = -1)  # 初始化一个Softmax操作
        self.to_qkv = nn.Linear(Dim, inner_dim * 3, bias = False)  # 对Q、K、V三组向量先进性线性操作

        # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, Dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的“头”数
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 先对Q、K、V进行线性操作，然后chunk乘三三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # 整理维度，获得Q、K、V

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子

        attn = self.attend(dots)  # 做Softmax运算

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
        return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out）


class Encoder1DBlock(nn.Module):
    """Transformer encoder 模块中的 layer结构."""
    def __init__(self,
                 Dim,
                 heads,
                 dim_head,
                 mlp_dim,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 **attention_kwargs
                ):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.
          mlp_dim: dimension of the mlp on top of attention block.
          dtype: the dtype of the computation (default: float32).
          dropout_rate: dropout rate.
          attention_dropout_rate: dropout for attention heads.
          deterministic: bool, deterministic or not (to apply dropout).
          **attention_kwargs: kwargs passed to nn.SelfAttention

        Returns:
          output after transformer encoder block.
        """
        super(Encoder1DBlock, self).__init__()
        self.Dim = Dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

    def forward(self, inputs):
        # Attention block.
        assert inputs.ndim == 3
        ln = nn.LayerNorm(self.Dim)
        x = ln(inputs)#torch.Size([B, 65, 1024])
        attention = Attention(Dim = self.Dim,
                            heads = self.heads,
                            dim_head = self.dim_head,
                            dropout = self.attention_dropout_rate
                            )
        x = attention(x)
        #print(x.shape)#torch.Size([B, 65, 1024])
        dropout = nn.Dropout(self.dropout_rate)
        x = dropout(x)#torch.Size([B, 65, 1024])
        x = x + inputs
        #print(x.shape)  # torch.Size([B, 65, 1024])

        # MLP block.
        ln = nn.LayerNorm(self.Dim)
        y = ln(x)
        #print(y.shape)#torch.Size([1, 65, 1024])
        mlpblock = MlpBlock(mlp_dim = self.mlp_dim,
                     Dim = self.Dim,
                     dropout_rate = self.dropout_rate)
        y = mlpblock(y)
        #print(y.shape)#torch.Size([B, 65, 1024])
        return x + y




class Encoder(nn.Module):
    """完整的Transformer Model Encoder 模块 for sequence to sequence translation."""

    def __init__(self,
                 Dim,
                 num_layers,
                 mlp_dim,
                 heads,
                 dim_head,
                 inputs_positions=None,
                 dropout_rate=0.1,
                 **attention_kwargs
                ):
        """Applies Transformer model on the inputs.
            Args:
              num_layers: number of layers
              mlp_dim: dimension of the mlp on top of attention block
              inputs_positions: input subsequence positions for packed examples.
              dropout_rate: dropout rate
              train: if it is training,
              **attention_kwargs: kwargs passed to nn.SelfAttention
            Returns:
              output of a transformer encoder.
        """
        super(Encoder, self).__init__()
        self.Dim = Dim
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.heads = heads
        self.dim_head = dim_head

    def forward(self,inputs):
        #inputs.shape = torch.Size([B, 65, 1024])

        assert inputs.ndim == 3  # (batch, len, emb)

        addpositionembs = AddPositionEmbs(inputs_positions = self.inputs_positions)
        x = addpositionembs(inputs)#torch.Size([B, 65, 1024]),添加了position embedding
        #print(x.shape)
        dropout = nn.Dropout(self.dropout_rate)
        x = dropout(x)#torch.Size([B, 65, 1024])

        # Input Encoder
        for lyr in range(self.num_layers):
            #添加num_layers个Encoder块构成完整的Transformer模块
            encoder1dblock = Encoder1DBlock(
                Dim = self.Dim,
                mlp_dim = self.mlp_dim,
                dropout_rate = self.dropout_rate,
                heads = self.heads,
                dim_head= self.dim_head
            )
            x = encoder1dblock(x)
        ln = nn.LayerNorm(self.Dim)
        encoded = ln(x)

        return encoded

class VisionTransformer(nn.Module):
    """完整的ViT结构"""
    
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,#一共有多少类别
                 Dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool = 'cls',
                 channels = 3,
                 dim_head = 64,
                 dropout = 0.,
                 emb_dropout = 0.1):
        super(VisionTransformer, self).__init__()
        image_height, image_width = pair(image_size)#image_size=256 -> image_height, image_width = 256
        patch_height, patch_width = pair(patch_size)#patch_size=32 -> patch_height, patch_width = 32

        #图像尺寸和patch尺寸必须要整除，否则报错
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #计算出一张图可以分成多少patch；这里：8*8=64，即一张图分成了64个patch
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width   #3*32*32=3*1024=3072，计算压成一维所需多少容量
        #print(patch_dim)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        #将一整张图变成patch embeding
        self.to_patch_embedding = nn.Sequential(
            #https://blog.csdn.net/csdn_yi_e/article/details/109143580
            #按给出的模式(注释)重组张量，其中模式中字母只是个表示，没有具体含义
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),#torch.Size([B, 64, 3072]),一张图像分成了64个patch，并将每个patch压成1维
            nn.Linear(patch_dim, Dim),#torch.Size([B, 64, 1024]),线性投影，将每个patch降维到1024维，得到patch embeddings
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, Dim))#torch.Size([1, 1, 1024])
        #print(self.cls_token.shape)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Encoder(num_layers = depth,#6
                                   mlp_dim = mlp_dim,#2048
                                   dropout_rate = dropout,#0.1
                                   heads = heads,#16
                                   dim_head = dim_head,#64
                                   Dim = Dim#1024
                                   )

        self.pool = pool
        self.to_latent = nn.Identity()#https://blog.csdn.net/artistkeepmonkey/article/details/115067356

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(Dim),
            nn.Linear(Dim, num_classes)##torch.Size([B, 1024]) -->
        )

    def forward(self, inputs):
        x = self.to_patch_embedding(inputs)
        #print(x.shape)
        b, n, _ = x.shape #torch.Size([B, 64, 1024])

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)#如果batch size=2，则torch.Size([2, 1, 1024])，相当于有多少张图就有多少个cls_token
        #print(cls_tokens.shape)#torch.Size([1, 1, 1024]),torch.Size([2, 1, 1024]),torch.Size(B, 1, 1024])....
        x = torch.cat((cls_tokens, x), dim=1)#torch.Size([B, 65, 1024])
        #print(x.shape)
        x = self.dropout(x)#torch.Size([B, 65, 1024])
        x = self.transformer(x)
        #向输入中添加position embedding -> Dropout -> num_layers个Encoder块 ->
            #对于每个Encoder块单元：输出都为：torch.Size([B, 65, 1024])
            #整个transformer输出尺寸：torch.Size([B, 65, 1024])

        #print(x[:, 0].shape)#torch.Size([B, 1024]),
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        #print(x.shape)#torch.Size([B, 1024])
        x = self.to_latent(x)#torch.Size([B, 1024])
        return self.mlp_head(x)#torch.Size([B, 1024]) --> torch.Size([B, num_classes])
        
