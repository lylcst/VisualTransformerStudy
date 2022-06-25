# -*-coding:utf-8-*-
# author lyl
import paddle
import paddle.nn as nn


class PatchEmbedding(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 dropout=0.):
        super(PatchEmbedding, self).__init__()

        self.num_batches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        # add class token
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.)
        )
        # add position embedding
        self.position_embedding = paddle.create_parameter(
            shape=[1, self.num_batches+1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=.02)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand([x.shape[0], -1, -1])
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1]) # [B, num_patches, embed_dim]
        x = paddle.concat([cls_token, x], axis=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None,
                             )
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])

        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn * self.scale)
        attn_weight = attn
        attn = self.attention_dropout(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, N, -1])

        out = self.proj(out)
        out = self.dropout(out)

        return out, attn_weight


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio), embed_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim=768,
                 num_heads=4,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super(EncoderLayer, self).__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, dropout, attention_dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x, attn_weight = self.attn(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class Encoder(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth=3,
                 qkv_bias=True,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super(Encoder, self).__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim, num_heads, qkv_bias, mlp_ratio, dropout, attention_dropout)
            layer_list.append(encoder_layer)
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class VisualTransformer(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=3,
                 num_heads=8,
                 qkv_bias=True,
                 mlp_ratio=4,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 num_classes=2):
        super(VisualTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, dropout)
        self.encoder = Encoder(embed_dim, num_heads, depth, qkv_bias, mlp_ratio, dropout, attention_dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [N, C, H, W]
        x = self.patch_embedding(x) # [N, num_patches, embed_dim]
        x = self.encoder(x)
        x = self.classifier(x[:, 0])

        return x


def main():
    vit = VisualTransformer()
    # print(vit)
    paddle.summary(vit, (4, 3, 224, 224))


if __name__ == '__main__':
    main()