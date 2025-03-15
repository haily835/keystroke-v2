import torch
from torch import nn
from torch_geometric.nn import MessagePassing, HypergraphConv
from einops import rearrange
import math
import numpy as np
from torchinfo import summary
import torch

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embed_dim, heads = heads, dim_head = dim_head),
                FeedForward(embed_dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, input_dim, embed_dim, depth, heads, mlp_dim, dim_head = 64):
        super().__init__()
        
        self.to_patch_embedding = nn.Sequential(
            # Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = 8,
            w = 1,
            dim = embed_dim,
        )
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim)
        self.pool = "mean"
        self.to_latent = nn.Identity()

        # self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return x
        # return self.linear_head(x)
    

    
def get_hi(batch_size, num_frames):
    vertices_per_graph = 42
    num_graphs = batch_size * num_frames
    edges_per_graph = 10

    # Given incidence matrix for one graph
    incidence_matrix_single = torch.tensor([
        [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=torch.long).T

    row_indices, col_indices = incidence_matrix_single.nonzero(as_tuple=True)
  
    row = row_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(row_indices)) * edges_per_graph
    col = col_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(col_indices)) * vertices_per_graph

    return torch.stack([col, row])

# Hypergraph convolution with temperal attention.
class unit_HCTA(nn.Module):
    def __init__(self, in_channels, n_joints, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.proj = nn.Linear(in_channels, out_channels)
        
        self.hc = HypergraphConv(in_channels=in_channels, out_channels=out_channels) # return [nodes, outfeatures]
        
        self.conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1,1), stride=(1,1))

        self.vit = SimpleViT(
            input_dim=out_channels * n_joints,
            embed_dim=out_channels * 8,
            depth=1,
            heads=4,
            mlp_dim=out_channels * n_joints,
        )

        self.fc = nn.Linear(out_channels * 8, out_channels * 8 * 21 * 2)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()

        # Residual
        res = rearrange(x, 'n c t v m -> n c (v m) t')
        res = self.bn(res)
        res = rearrange(res, 'n c vm t -> n t vm c')
        res = self.proj(res)
        res = rearrange(res, 'n t vm c -> n t (vm c)')
        # print("Residue ", res.shape)

        x = rearrange(x, 'n c t v m -> (n t v m) c')
        
        x = self.hc(x, hi)
        # print("After hyperconv", x.shape)
        
        x = x.view(N, -1, T, V, M)
        x = rearrange(x, 'n c t v m -> n t (v m) c')
        # print("Before conv across node feature", x.shape)
        x = self.conv(x)
        # print("After conv", x.shape)
        # x = x.view(N, -1, T, V, M)
        x = rearrange(x, 'n t vm c -> n t (c vm)')
        
        # print("Input for attention", x.shape)
        
        x = self.vit(x)
        # print("After temperal attention", x.shape)

        x = self.fc(x)
        # print("After FC", x.shape)
        x = x.view(N, T, -1)
        x = self.act(x + res)
       
        # Reshape back the output to match the batch size
        x = x.view(N, -1, T, V, M)

        # print("Finish ", x.shape)
        return x

class HCTA(nn.Module):
    def __init__(self, num_class=40, n_joints=42, num_frames=8):
        super().__init__()

        # print(self.hi)
        self.l1=unit_HCTA(3, n_joints, 4)
        self.l2=unit_HCTA(4, n_joints, 4)
        self.l3=unit_HCTA(4, n_joints, 4)
        self.l4=unit_HCTA(4, n_joints, 8)
        self.l5=unit_HCTA(8, n_joints, 8)
        self.l6=unit_HCTA(8, n_joints, 8)
        self.l7=unit_HCTA(8, n_joints, 8)
        
        # self.l2=unit_HCTA(8, n_joints, 8)
        # self.l3=unit_HCTA(8, n_joints, 16)

        
        # self.l4=unit_HCTA(16, n_joints, 16)
        # self.l5=unit_HCTA(16, n_joints, 16)
        # self.l6=unit_HCTA(16, n_joints, 32)

        # self.l7=HCTA(32, n_joints, 32)
        # self.l8=HCTA(32, n_joints, 32)
        # self.l9=HCTA(32, n_joints, 32)
        

        self.flat = nn.Flatten(1)
        self.fc = nn.Linear(8*n_joints*num_frames, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        
        hi = get_hi(N, T).to(x.device)
        x = self.l1(x, hi)
        x = self.l2(x, hi)
        x = self.l3(x, hi)
        x = self.l4(x, hi)
        
        x = self.l5(x, hi)
        x = self.l6(x, hi)
        x = self.l7(x, hi)
        # x = self.l8(x, hi)
        # x = self.l9(x, hi)
        
        x = self.flat(x)
        x = self.fc(x)
       
        return x
    

if __name__ == "__main__":
    x = torch.rand((32, 3, 8, 21, 2))
    model = HCTA(num_class=40,n_joints=42, num_frames=8)
 
    print(summary(model, (32, 3, 8, 21, 2)))
    print(model(x).shape)