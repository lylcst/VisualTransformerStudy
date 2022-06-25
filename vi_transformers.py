import paddle
import paddle.nn as nn


# paddle.set_device('cpu')

class PatchEmbedding(nn.Layer):
    def __init__(self,
                image_size,
                patch_size,
                in_channels,
                embed_dim,
                dropout=0.):
        super().__init__()
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                        out_channels=embed_dim,
                                        kernel_size=patch_size,
                                        stride=patch_size)
        self.dropout = nn.Dropout(dropout)

        self.class_token = paddle.create_parameter(
            shape=[1,1,embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(0.)
        )
        self.position_embedding = paddle.create_parameter(
            shape=[1, n_patches+1, embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.0)
        )
    
    def forward(self, x):
        cls_token = self.class_token.expand([x.shape[0], -1, -1])
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1]) #[n, num_patches, embed_dim]
        x = paddle.concat([cls_token, x], axis=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        return x


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.) -> None:
        super().__init__()
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


class Attention(nn.Layer):
    def __init__(self,
                embed_dim,
                num_heads,
                qkv_bias=True,
                dropout=0.,
                attention_dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim*3,
                             bias_attr=False if not qkv_bias else None)

        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])

        return x


    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multihead, qkv)

        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, N, -1])

        out = self.dropout(out)
        return out
        

class EncoderLayer(nn.Layer):
    def __init__(self,
                embed_dim,
                num_heads,
                qkv_bias=True,
                mlp_ratio=4.0,
                dropout=0.,
                attention_dropout=0.) -> None:
        super().__init__()
        self.attention = Attention(embed_dim=embed_dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   dropout=dropout,
                                   attention_dropout=attention_dropout)
        self.mlp = Mlp(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(attention_dropout)

        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attention(x)
        x = x + h
        x = self.attn_dropout(x)

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x =  x + h
        x = self.mlp_dropout(x)

        return x


class Encoder(nn.Layer):
    def __init__(self,
                image_size,
                patch_size,
                in_channels=3,
                embed_dim=768,
                num_heads=12,
                num_layers=6,
                qkv_bias=True,
                mlp_ratio=4.0,
                dropout=0.,
                attention_dropout=0.) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              in_channels=in_channels,
                                              embed_dim=embed_dim,
                                              dropout=dropout)
        self.encoder = nn.LayerList([EncoderLayer(embed_dim, num_heads, qkv_bias, mlp_ratio, dropout, attention_dropout) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [N, C, H, W]
        x = self.patch_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.dropout(x)

        return x


class VisualTransformer(nn.Layer):
    def __init__(self,
                image_size=224,
                patch_size=14,
                in_channels=3,
                embed_dim=768,
                num_heads=12,
                num_layers=6,
                qkv_bias=True,
                mlp_ratio=4.0,
                num_classes=2,
                dropout=0.,
                attention_dropout=0.) -> None:
        super().__init__()
        self.encoder = Encoder(image_size,
                               patch_size,
                               in_channels,
                               embed_dim,
                               num_heads,
                               num_layers,
                               qkv_bias,
                               mlp_ratio,
                               dropout,
                               attention_dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        # self.softmax = nn.Softmax(axis=-1)
    
    def forward(self, x, label=None):
        x = self.encoder(x)
        x = self.classifier(x)
        # x = self.softmax(x)

        return x


def main():
    vit = VisualTransformer()
    paddle.summary(vit, (4, 3, 224, 224))
    # patch_embedding = PatchEmbedding(image_size=28,
    #                                 patch_size=7,
    #                                 in_channels=3,
    #                                 embed_dim=768)
    # mlp = Mlp(embed_dim=768)
    # attention = Attention(embed_dim=768, num_heads=12)
    # t = paddle.randn([4, 3, 28, 28])
    # out = patch_embedding(t)
    # out = mlp(out)
    # out = attention(out)

    # print(out.shape)

if __name__ == '__main__':
    main()
