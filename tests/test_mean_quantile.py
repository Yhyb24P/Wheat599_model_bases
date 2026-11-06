import pytest
import torch
from wheat599.mean_quantile import mean_loss


def test_member_mean_supervision_and_gradient():
    q=torch.tensor([[[0.,2.,4.],[2.,4.,6.]]],requires_grad=True)
    y=torch.tensor([[1.]])
    loss=mean_loss(q,y,.1)
    assert loss.item()==pytest.approx(.1*3*(1+9)/2)
    loss.backward()
    torch.testing.assert_close(q.grad,torch.tensor([[[.1,.1,.1],[.3,.3,.3]]]))
    assert mean_loss(q,y,0).item()==0
