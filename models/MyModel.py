import torch
from torch import nn
from torch_geometric.nn import MessagePassing, GATv2Conv
from einops import rearrange
import math
import numpy as np
from torchinfo import summary

import torch
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels) 
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

def get_hi(batch_size, num_frames):
    vertices_per_graph = 21
    num_graphs = batch_size * num_frames * 2
    edges_per_graph = 6

    # Given incidence matrix for one graph
    incidence_matrix_single = torch.tensor([
        [1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1]
    ], dtype=torch.long).T

    row_indices, col_indices = incidence_matrix_single.nonzero(as_tuple=True)
  
    row = row_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(row_indices)) * edges_per_graph
    col = col_indices.repeat(num_graphs) + torch.arange(num_graphs).repeat_interleave(len(col_indices)) * vertices_per_graph

    return torch.stack([col, row])

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super(unit_gcn, self).__init__()    
        self.convs = nn.ModuleList()
        for i in range(5):
            self.convs.append(GATv2Conv(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        NM, C, T, V = x.size()
        hi = get_hi(NM // 2, T).to(x.device)
        reshaped = rearrange(x, 'nm c t v -> (nm t v) c')
        y = None
        for i in range(3):
            z = self.convs[i](reshaped, hi)
            y = z + y if y is not None else z
        
        y = y.view(NM, -1, T, V)

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y

class TCN_HC_unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True, kernel_size=5, dilations=[1,2]):
        super().__init__()
        self.hc = unit_gcn(in_channels=in_channels, out_channels=out_channels, residual=True)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           dilations=dilations,residual=False)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        NM, C, T, V = x.size()

        res = self.residual(x)
       
        x = self.hc(x)

        x = x.view(NM, -1, T, V)
        x = self.tcn(x)
        x = self.relu(x + res)
        return x

class MyModel(nn.Module):
    def __init__(self, num_class=40, num_point=21, num_person=2, in_channels=3,
                 drop_out=0):
        super(MyModel, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 48
        self.l1 = TCN_HC_unit(in_channels, base_channel, residual=False)
        self.l2 = TCN_HC_unit(base_channel, base_channel)
        self.l3 = TCN_HC_unit(base_channel, base_channel)
        self.l4 = TCN_HC_unit(base_channel, base_channel)
        self.l5 = TCN_HC_unit(base_channel, base_channel*2, stride=2)
        self.l6 = TCN_HC_unit(base_channel*2, base_channel*2)
        self.l7 = TCN_HC_unit(base_channel*2, base_channel*2)
        self.l8 = TCN_HC_unit(base_channel*2, base_channel*4, stride=2)
        self.l9 = TCN_HC_unit(base_channel*4, base_channel*4)
        self.l10 = TCN_HC_unit(base_channel*4, base_channel*4)
        # self.l11 = TCN_HC_unit(base_channel*4, base_channel*4)
        # self.l12 = TCN_HC_unit(base_channel*4, base_channel*4)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
       
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        # x = self.l11(x)
        # x = self.l12(x)
        # print(x.shape)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

if __name__ == "__main__":
    # x = torch.rand((32, 3, 8, 21, 2))
    model = MyModel()
    print(summary(model, (32, 3, 8, 21, 2)))
    # x = model(x)