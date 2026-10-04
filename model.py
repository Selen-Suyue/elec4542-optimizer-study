"""Fixed, randomly initialized ViT for the ELEC4542 optimizer study."""
import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):

    def __init__(self, dim=192, heads=3):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(z.transpose(1, 2).reshape(b, n, d))

class TransformerBlock(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SmallViT(nn.Module):

    def __init__(self, classes=200, dim=192, depth=6, heads=3, patch=8):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, (64 // patch) ** 2 + 1, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.patch(x).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(x.shape[0], -1, -1), z], 1) + self.pos
        return self.head(self.norm(self.blocks(z))[:, 0])
