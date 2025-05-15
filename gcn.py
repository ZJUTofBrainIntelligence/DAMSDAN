import torch
import torch.nn.functional as F
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def mish(x):
    return x*(torch.tanh(F.softplus(x)))

class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio=40):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias=False),
                                nn.ELU(inplace=False),
                                nn.Linear(inc // reduction_ratio, inc, bias=False),
                                nn.Tanh(),
                                nn.ReLU(inplace=False))

    def forward(self, x):
        y = self.fc(x)
        return y

class GCN(nn.Module):
    def __init__(self,  in_channels=8, node_num=19, batchsize=32, **kwargs):
        super().__init__()
        # build networks
        self.bs = batchsize
        spatial_kernel_size = 1
        kernel_size = (1, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * node_num)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.gcn_networks = nn.ModuleList((

            gcn(in_channels, in_channels, kernel_size, residual=False, **kwargs0),
            gcn(in_channels, in_channels, kernel_size, **kwargs),
            gcn(in_channels, in_channels, kernel_size, **kwargs),
        ))
        self.chan_num = node_num
        self.A = torch.rand((self.bs, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False).cuda()
        self.GATENet = GATENet(self.chan_num * self.chan_num, reduction_ratio=40)


    def forward(self, x):
        A_ds = self.GATENet(self.A)
        adj = A_ds.reshape(self.bs, self.chan_num, self.chan_num)
        adj = adj.reshape(self.bs, -1, self.chan_num, self.chan_num)
        N, C, T, V = x.size()
        x = x.transpose(2,3).contiguous()    # N, C, V, T
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).transpose(2, 3).contiguous()    # N, C, T, V

        #forward
        for i, gcn in enumerate(self.gcn_networks):
            x = gcn(x, adj)

        return x

class gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,       # 时间维度上的感知野,邻接矩阵的分区数
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        x, A = self.gcn(x, A)
        return self.relu(x)

class ConvTemporalGraphical(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))
        return x.contiguous(), A

if __name__ == '__main__':
    x1 = torch.rand(32, 3, 114, 32).to(device)
    mo = GCN().to(device)
    y = mo(x1)
    print(y.shape)