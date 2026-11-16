"""Member-conditioned marker interactions injected into the official TabM blocks."""
import math

import torch
from torch import nn


class MemberInteractionBackbone(nn.Module):
    def __init__(self, backbone, markers, rank=32):
        super().__init__()
        self.base=backbone
        first=backbone.backbone.blocks[0][0]
        members,width=first.r.shape[0],first.weight.shape[0]
        if first.r.shape!=(members,markers):raise ValueError('Unexpected first-layer member scaling')
        self.factors=nn.Parameter(torch.randn(markers,rank)/math.sqrt(markers))
        self.keys=nn.ModuleList([nn.Linear(rank,width,bias=False) for _ in backbone.backbone.blocks])
        self.values=nn.ModuleList([nn.Linear(rank,width,bias=False) for _ in backbone.backbone.blocks])
        self.norm=nn.RMSNorm(width,elementwise_affine=False)
        self.alpha=nn.Parameter(torch.zeros(len(backbone.backbone.blocks)))

    def forward(self,z):
        first=self.base.backbone.blocks[0][0]
        scaled=z[:,None,:]*first.r[None,:,:]
        projected=scaled@self.factors
        interactions=.5*(projected.square()-scaled.square()@self.factors.square())
        h=self.base.ensemble_view(z)
        for i,(block,key,value) in enumerate(zip(self.base.backbone.blocks,self.keys,self.values)):
            u=block(h)
            gate=torch.sigmoid((self.norm(u)*self.norm(key(interactions))).sum(dim=-1,keepdim=True)/math.sqrt(u.shape[-1]))
            h=u+self.alpha[i]*gate*value(interactions)
        return self.base.output(h)
