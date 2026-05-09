import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import ConvNextBlock

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=4):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class BasicMotionEncoder(nn.Module):
    def __init__(self, args, dim=128):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_channel
        self.convc1 = nn.Conv2d(cor_planes, dim*2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim*2, dim+dim//2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim//2, 3, padding=1)
        self.conv = nn.Conv2d(dim*2, dim-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hdim=128, cdim=128):
        #net: hdim, inp: cdim
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args, dim=cdim)
        self.refine = []
        for i in range(args.num_blocks):
            self.refine.append(ConvNextBlock(2*cdim+hdim, hdim))
        self.refine = nn.ModuleList(self.refine)

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net