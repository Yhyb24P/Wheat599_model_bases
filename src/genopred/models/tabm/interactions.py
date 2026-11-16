"""TabM marker interactions, member connections and optimizer groups."""
import math

import torch
from torch import nn

from genopred.models.tabm.quantile import quantile_loss


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


class LayerInteractionBackbone(nn.Module):
    """Read the same marker-interaction basis in each residual block."""
    def __init__(self, markers, *, width, members, dropout, quantiles, rank=32, blocks=3):
        super().__init__()
        import tabm
        self.factors=nn.Parameter(torch.randn(markers,rank)/math.sqrt(markers))
        self.additive=nn.Linear(markers,width)
        self.member_input=nn.Parameter(torch.empty(members,markers).bernoulli_(.5).mul_(2).sub_(1))
        self.member_output=nn.Parameter(torch.ones(members,width))
        self.blocks=tabm.TabM.make(n_num_features=width,d_out=quantiles,n_blocks=blocks,
                                  d_block=width,dropout=dropout,k=members)
        self.pair_norm=nn.RMSNorm(rank)
        self.norms=nn.ModuleList([nn.RMSNorm(width) for _ in range(blocks)])
        self.project=nn.ModuleList([nn.Linear(rank,width) for _ in range(blocks)])
        self.gates=nn.ModuleList([nn.Linear(width+rank,width) for _ in range(blocks)])

    def forward(self,z):
        projected=z@self.factors
        pair=self.pair_norm(.5*(projected.square()-z.square()@self.factors.square()))
        h=self.additive(z[:,None,:]*self.member_input[None,:,:])*self.member_output[None,:,:]
        m=pair[:,None,:].expand(-1,h.shape[1],-1)
        for block,norm,project,gate in zip(self.blocks.backbone.blocks,self.norms,self.project,self.gates):
            h=h+torch.sigmoid(gate(torch.cat((norm(h),m),dim=-1)))*project(m)
            h=h+block(norm(h))
        return self.blocks.output(h)


def make_optimizers(model, config, variant):
    if variant=='adam':
        return [torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])]
    # Only hidden BatchEnsemble matrices: member scaling, gates and outputs use AdamW.
    blocks=model.backbone.blocks.backbone.blocks if variant in ('gated','layered') else model.backbone.backbone.blocks
    muon=[block[0].weight for block in blocks]
    ids={id(p) for p in muon}
    other=[p for p in model.parameters() if id(p) not in ids]
    if not muon or len(ids)!=len(muon):raise ValueError('Invalid Muon parameter groups')
    return [torch.optim.Muon(muon,lr=config['muon_lr'],weight_decay=config['weight_decay']),
            torch.optim.AdamW(other,lr=config['learning_rate'],weight_decay=config['weight_decay'])]


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


class ConnectionBasis(nn.Module):
    def __init__(self, base, rank, *, shared=True):
        super().__init__()
        if not isinstance(rank, int) or rank < 1:
            raise ValueError('Connection rank must be positive')
        self.base, self.shared = base, shared
        members, markers = base.r.shape
        width = base.weight.shape[0]
        shape = () if shared else (members,)
        self.u = nn.Parameter(torch.randn(*shape, width, rank, dtype=base.weight.dtype, device='cpu'))
        self.v = nn.Parameter(torch.randn(*shape, markers, rank, dtype=base.weight.dtype, device='cpu')
                              if shared else torch.zeros(members, markers, rank, dtype=base.weight.dtype, device='cpu'))
        if shared:
            self.coefficients = nn.Parameter(torch.zeros(members, rank, dtype=base.weight.dtype, device='cpu'))

    def delta(self):
        if self.shared:
            c = self.coefficients - self.coefficients.mean(0, keepdim=True)
            modulation = (c[:, None, :] * self.u[None]) @ self.v.T
            return self.base.weight[None] * modulation
        change = self.base.weight[None] * (self.u @ self.v.transpose(-1, -2))
        # Center actual changes; independent factors need not share directions.
        return change - change.mean(0, keepdim=True)

    def penalty(self):
        return self.delta().square().mean()

    def forward(self, x):
        if self.shared:
            c = self.coefficients - self.coefficients.mean(0, keepdim=True)
            # GenomicTabM repeats the same standardized markers across members.
            g = torch.nn.functional.linear(x[:, :1, :] * self.v.T[None], self.base.weight)
            extra = torch.einsum('brh,mr->bmh', g * self.u.T[None], c)
        else:
            extra = torch.einsum('bmp,mhp->bmh', x, self.delta())
        return self.base(x) + extra


def basis_loss(model, x, y, quantiles, strength):
    loss = quantile_loss(model(x), y, quantiles)
    if strength:
        loss = loss + strength * model.backbone.backbone.blocks[0][0].penalty()
    return loss
