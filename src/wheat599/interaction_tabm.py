"""Full-marker bi-interactions and four-stream gated TabM backbone."""
import math

import torch
from torch import nn


class InteractionBackbone(nn.Module):
    def __init__(self, markers, *, width, members, dropout, quantiles, rank=32, blocks=3):
        super().__init__()
        import tabm
        self.factors=nn.Parameter(torch.randn(markers,rank)/math.sqrt(markers))
        self.additive=nn.Linear(markers,width)
        self.interaction=nn.Linear(rank,width)
        self.member_input=nn.Parameter(torch.empty(members,markers).bernoulli_(.5).mul_(2).sub_(1))
        self.member_output=nn.Parameter(torch.ones(members,width))
        self.blocks=tabm.TabM.make(n_num_features=width,d_out=quantiles,n_blocks=blocks,
                                  d_block=width,dropout=dropout,k=members)
        self.norms=nn.ModuleList([nn.RMSNorm(width) for _ in range(blocks)])
        self.read=nn.ModuleList([nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,4*width)) for _ in range(blocks)])
        self.write=nn.ModuleList([nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,4)) for _ in range(blocks)])

    def forward(self,z):
        projected=z@self.factors
        pair=.5*(projected.square()-z.square()@self.factors.square())
        a=self.additive(z[:,None,:]*self.member_input[None,:,:])*self.member_output[None,:,:]
        b=self.interaction(pair)[:,None,:].expand_as(a)
        h=torch.stack((a,b,(a+b)/2,(a-b)/2),dim=2)
        for block,norm,read,write in zip(self.blocks.backbone.blocks,self.norms,self.read,self.write):
            context=h.mean(dim=2)
            rg=torch.sigmoid(read(context)).reshape(*context.shape[:2],4,context.shape[-1])
            wg=torch.sigmoid(write(context)).unsqueeze(-1)
            r=(rg*norm(h)).mean(dim=2)
            h=h+wg*block(r).unsqueeze(2)
        return self.blocks.output(h.mean(dim=2))


def make_optimizers(model, config, variant):
    if variant=='adam':
        return [torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])]
    # Only hidden BatchEnsemble matrices: member scaling, gates and outputs use AdamW.
    blocks=model.backbone.blocks.backbone.blocks if variant=='gated' else model.backbone.backbone.blocks
    muon=[block[0].weight for block in blocks]
    ids={id(p) for p in muon}
    other=[p for p in model.parameters() if id(p) not in ids]
    if not muon or len(ids)!=len(muon):raise ValueError('Invalid Muon parameter groups')
    return [torch.optim.Muon(muon,lr=config['muon_lr'],weight_decay=config['weight_decay']),
            torch.optim.AdamW(other,lr=config['learning_rate'],weight_decay=config['weight_decay'])]
