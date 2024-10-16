import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]796/8
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_size, num_patches, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        # 位置编码，位置编码矩阵的维度是(1, num_patches + 1, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim), requires_grad=True)
        # 类别嵌入，类别嵌入矩阵的维度是(1, 1, embedding_dim)， 用于分类任务
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout=0.1):
        super(MLPClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_blocks, nhead, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.layer = nn.ModuleList()
        self.TransformerLayer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            MultiHeadAttention(embedding_dim, nhead, dropout=dropout)
        )
        self.FeedForward = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        for _ in range(num_blocks):
            self.layer.append(nn.ModuleList([self.TransformerLayer, self.FeedForward]))

    def forward(self, x):
        for transformer, feedforward in self.layer:
            x = transformer(x) + x
            x = feedforward(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, embedding_dim, num_blocks, nhead, num_classes, patch_size, num_patches, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embedding_dim, patch_size, num_patches, dropout=dropout)
        self.Transformer = TransformerBlock(embedding_dim, num_blocks, nhead, dropout=dropout)
        self.mlp_classifier = MLPClassifier(embedding_dim, num_classes, dropout=dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.Transformer(x)
        x = self.mlp_classifier(x[:, 0])
        return x


if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
    ViT = ViT(3, 768, 12, 8, 10, 16, 196)
    output = ViT(input)
    print(output.shape)
