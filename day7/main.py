import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image


paddle.set_device('cpu')


class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = x.transpose([0, 2, 1])
        x = self.norm(x)

        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B*num_patches, ws, ws, c]
    x = x.reshape([-1, window_size, window_size, C])
    return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H / window_size * W / window_size))
    x = windows.reshape([B, H//window_size, W//window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x


class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scales = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x, mask=None):
        # x: [B, num_patches, embed_dim]
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scales
        attn = paddle.matmul(q, k, transpose_y=True)
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows, num_patches, num_patches]
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
            attn = attn.reshape([B//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[2]])
            # attn: [B, num_windows, num_heads, num_patches, num_patches]
            # mask: [1, num_windows, 1,         num_patches, num_patches]
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])

        # [B, num_heads, num_patches, head_dim]
        out = paddle.matmul(attn, v)
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, N, -1])
        out = self.proj(out)

        return out


class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.resolution = input_resolution
        self.window_size = window_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size, input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape

        h = x
        x = self.attn_norm(x)
        # TODO
        x = x.reshape([B, H, W, C])
 
        # TODO: shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # TODO: compute window attn
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # shift back
        if self.shift_size > 0:
                x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        # [B, H, W, C]
        x = x.reshape([B, H * W, -1])
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


# TODO generate attn mask
def generate_mask(window_size=4, shift_size=2, input_resolution=(8, 8)):
    H, W = input_resolution
    img_mask = paddle.zeros([1, H, W, 1])
    # img_mask = np.zeros([1, H, W, 1])
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w,:] = cnt
            cnt += 1

    windows_mask = windows_partition(img_mask, window_size=window_size)
    windows_mask = windows_mask.reshape([-1, window_size * window_size])

    attn_mask = windows_mask.unsqueeze(1) - windows_mask.unsqueeze(2)
    attn_mask = paddle.where(attn_mask != 0, paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
                        
    return attn_mask


def main():
    # img_mask = generate_mask()
    # print(img_mask.squeeze())
    # img_mask = img_mask.cpu().numpy().astype('uint8')
    # for i in range(4):
    #     for j in range(16):
    #         for k in range(16):
    #             print(img_mask[i, j, k], end='\t')
    #         print()

    #     print()
    #     im = Image.fromarray(img_mask[i, :, :])
    #     im.save(f"{i}.png")
    #     print()
    # print()
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    out = patch_embedding(t)
    print(out.shape)
    out = swin_block(out)
    print(out.shape)
    out = patch_merging(out)
    print(out.shape)


if __name__ == '__main__':
    main()








