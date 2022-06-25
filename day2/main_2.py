# -*-coding:utf-8-*-
# author lyl
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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


class PatchEmbedding(nn.Layer):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (image_size // patch_size) * 2
        self.embed_dim = embed_dim
        self.patch_embedding = nn.Conv2D(in_channels,
                                         embed_dim,
                                         patch_size,
                                         stride=patch_size,
                                         bias_attr=False)
        self.dropout = nn.Dropout(dropout)

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.)
        )

        # add position embedding
        self.position_embedding = paddle.create_parameter(
            shape=[1, self.n_patches+1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=0.02)
        )

    def forward(self, x):
        # x: [1, 1, 28, 28]
        cls_tokens = self.cls_token.expand([x.shape[0], 1, self.embed_dim])
        x = self.patch_embedding(x)
        # x: [1, embed_dim, image_size/patch_size, image_size/patch_size]
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = paddle.concat([cls_tokens, x], axis=1)
        x = self.dropout(x)
        x = x + self.position_embedding
        return x


class Attention(nn.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x):
        return x


class Encoder(nn.Layer):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.attn = Attention()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


class ViT(nn.Layer):
    def __init__(self):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(image_size=224,
                                          patch_size=7,
                                          in_channels=3,
                                          embed_dim=16)
        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.LayerList(layer_list)
        self.head = nn.Linear(16, 10)
        self.avgpool = nn.AdaptiveAvgPool1D(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)
        # layernorm
        # [n, h*w, c]
        x = x.transpose([0, 2, 1])
        x = self.avgpool(x) # [n, c, 1]
        x = x.flatten(1) # [n, c]
        x = self.head(x)
        return x


def main():
    # 1. Load image and convert to tensor
    # img = Image.open('')
    # img = np.array(img)
    img = np.random.randint(0, 10, [28, 28])
    # for i in range(28):
    #     for j in range(28):
    #         print(f"{img[i, j]:03}", end=' ')
    #     print()

    sample = paddle.to_tensor(img, dtype='float32')
    # simulate a batch of data
    sample = sample.reshape([1, 1, 28, 28])
    print(sample.shape)

    # 2. Patch Embedding
    patch_embed = PatchEmbedding(image_size=28,
                                 patch_size=7,
                                 in_channels=1,
                                 embed_dim=1)
    out = patch_embed(sample)
    print(out)
    # for i in range(0, 28, 7):
    #     for j in range(0, 28, 7):
    #         print(paddle.sum(sample[0, 0, i:i+7, j:j+7]).numpy().item())
    # 3. Mlp
    mlp = Mlp(embed_dim=1)
    out = mlp(out)
    print(out.shape)


if __name__ == '__main__':
    # main()
    t = paddle.randn([4, 3, 224, 224])
    model = ViT()
    out = model(t)
    print(out)