import paddle
import paddle.nn as nn
from resnet import ResNet18


paddle.set_device('cpu')


class PositionEmbedding(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.row_embed = nn.Embedding(50, embed_dim)
        self.col_embed = nn.Embedding(50, embed_dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        i = paddle.arange(w)
        j = paddle.arange(h)
        x_embed = self.col_embed(i)
        y_embed = self.row_embed(j)
        pos = paddle.concat([x_embed.unsqueeze(0).expand((h, x_embed.shape[0], x_embed.shape[1])),
                             y_embed.unsqueeze(1).expand((y_embed.shape[0], w, y_embed.shape[1]))])
        pos = pos[:pos.shape[0]//2, :, :]
        pos = pos.transpose([2, 0, 1])
        pos = pos.unsqueeze(0)
        pos = pos.expand([x.shape[0]] + pos.shape[1::])
        return pos


class Mlp(nn.Layer):
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Layer):
    def __init__(self, embed_dim=768, num_head=8, dropout=0., attention_dropout=0.):
        super().__init__()
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.all_head_dim = self.head_dim * self.num_head
        self.scales = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, self.all_head_dim)
        self.k_proj = nn.Linear(embed_dim, self.all_head_dim)
        self.v_proj = nn.Linear(embed_dim, self.all_head_dim)

        self.proj = nn.Linear(self.all_head_dim, embed_dim)
        self.softmax = nn.Softmax(axis=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_head, self.head_dim]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, q, k, v):
        bz, N, _ = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q, k, v = map(self.transpose_multihead, [q, k, v])
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = attn * self.scales
        attn = self.softmax(attn)
        attn = self.attention_dropout(attn)

        out = paddle.matmul(attn, v) # [bz, num_head, num_patches, embed]
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([bz, N, -1])
        out = self.dropout(out)

        return out


class EncoderLayer(nn.Layer):
    def __init__(self, 
                 embed_dim=768, 
                 num_head=8, 
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_head, dropout, attention_dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)
    
    def forward(self, x, pos=None):
        h = x
        x = self.attn_norm(x)
        q = x + pos if pos is not None else x
        k = x + pos if pos is not None else x
        out = self.attn(q, k, x)
        x = h + out

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class DecoderLayer(nn.Layer):
    def __init__(self, 
                 embed_dim=768, 
                 num_head=4, 
                 mlp_ratio=4.,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, 
                              num_head,
                              attention_dropout,
                              dropout)
        self.enc_dec_attn_norm = nn.LayerNorm(embed_dim)
        self.enc_dec_attn = Attention(embed_dim,
                                 num_head,
                                 attention_dropout,
                                 dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio)
    
    def forward(self, x, enc_out, query_pos=None, pos=None):
        h = x
        x = self.attn_norm(x)
        q = x + query_pos if query_pos is not None else x
        k = x + query_pos if query_pos is not None else x
        x = self.attn(q, k, x)
        x = x + h

        h = x
        x = self.enc_dec_attn_norm(x)
        q = x + query_pos if query_pos is not None else x
        k = enc_out + pos if pos is not None else enc_out
        x = self.enc_dec_attn(q, k, enc_out)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class Encoder(nn.Layer):
    def __init__(self,
                 embed_dim=768, 
                 num_head=8, 
                 num_layer=6,
                 mlp_ratio=4.0,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.encoders = nn.LayerList([
            EncoderLayer(embed_dim, num_head, mlp_ratio, attention_dropout, dropout) 
            for i in range(num_layer)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None):
        for encoder_layer in self.encoders:
            x = encoder_layer(x, pos)
        x = self.dropout(x)
        return x


class Decoder(nn.Layer):
    def __init__(self, 
                 embed_dim=768,
                 num_head=4,
                 num_layer=6,
                 mlp_ratio=4.,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.decoders = nn.LayerList([
            DecoderLayer(embed_dim, 
                         num_head, 
                         mlp_ratio, 
                         attention_dropout, 
                         dropout)
            for i in range(num_layer)
        ])

    def forward(self, x, enc_out, query_pos=None, pos=None):
        for decoder_layer in self.decoders:
            x = decoder_layer(x, enc_out, query_pos, pos)
        return x


class Transformer(nn.Layer):
    def __init__(self, 
                 embed_dim=768, 
                 num_head=4, 
                 num_layer=6,
                 mlp_ratio=4.,
                 attention_dropout=0.,
                 dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(embed_dim,
                               num_head,
                               num_layer,
                               mlp_ratio,
                               attention_dropout,
                               dropout)
        self.decoder = Decoder(embed_dim,
                               num_head,
                               num_layer,
                               mlp_ratio,
                               attention_dropout,
                               dropout)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, query_embed, pos_embed):
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose([0, 2, 1]) # [B, H*W, C] -> [B, num_patches,C]

        # [B, dim, H, W]
        pos_embed = pos_embed.flatten(2)
        pos_embed = pos_embed.transpose([0, 2, 1])

        # [num_queries, dim]
        query_embed = query_embed.unsqueeze(0)
        query_embed = query_embed.expand((B, query_embed.shape[1], query_embed.shape[2]))

        target = paddle.zeros_like(query_embed)

        x = self.encoder(x, pos_embed)
        encoder_out = self.encoder_norm(x)

        decoder_out = self.decoder(target, encoder_out, query_embed, pos_embed)
        decoder_out = self.decoder_norm(decoder_out)
        print(f"---decoder out: {decoder_out.shape}")
        
        return decoder_out


class BboxEmbed(nn.Layer):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x



class DETR(nn.Layer):
    def __init__(self, backbone, pos_embed, transformer, num_classes, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        embed_dim = transformer.embed_dim

        self.class_embed = nn.Linear(embed_dim, num_classes+1)
        self.bbox_embed = BboxEmbed(embed_dim, embed_dim, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.input_proj = nn.Conv2D(backbone.num_channels, embed_dim, kernel_size=1)
        self.backbone = backbone
        self.pos_embed = pos_embed

    def forward(self, x):
        print(f'--INPUT: {x.shape}')
        feat = self.backbone(x)
        print(f'--Feature after ResNet18: {feat.shape}')
        feat = self.input_proj(feat)
        pos_embed = self.pos_embed(feat)

        out= self.transformer(feat, self.query_embed.weight, pos_embed)

        out_class = self.class_embed(out)
        out_coord = self.bbox_embed(out)

        return out_class, out_coord



def build_detr():
    backbone = ResNet18()
    transformer = Transformer()
    pos_embed = PositionEmbedding(768)
    detr = DETR(backbone, pos_embed, transformer, 10, 100)
    return detr


def main():
    t = paddle.randn([4, 3, 224, 224])
    model = build_detr()
    out = model(t)
    print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    main()