import torch

from joatmon.nn import functional
from joatmon.nn.loss.huber import HuberLoss

a = torch.rand((2, 2), requires_grad=True)
x = torch.ones((2, 2), requires_grad=True)
y = torch.zeros((2, 2), requires_grad=True)

z = torch.where(a > 0.5, x, y)

huber = torch.nn.HuberLoss()
loss = huber(z, torch.zeros_like(z))

loss.backward()

print(f'{a=:}, {a.requires_grad=:}, {a.grad_fn=:}, {a.grad}')
print(f'{x=:}, {x.requires_grad=:}, {x.grad_fn=:}, {x.grad}')
print(f'{y=:}, {y.requires_grad=:}, {y.grad_fn=:}, {y.grad}')
print(f'{z=:}, {z.requires_grad=:}, {z.grad_fn=:}, {z.grad}')

print('-' * 30)

import joatmon.nn.functional as f

a = f.from_array(a.detach().numpy(), requires_grad=True)
x = f.ones((2, 2), requires_grad=True)
y = f.zeros((2, 2), requires_grad=True)

z = f.where(a > 0.5, x, y)

huber = HuberLoss()
loss = huber(z, functional.zeros_like(z))

loss.backward()

print(f'{a=:}, {a.requires_grad=:}, {a._grad_fn=:}, {a.grad}')
print(f'{x=:}, {x.requires_grad=:}, {x._grad_fn=:}, {x.grad}')
print(f'{y=:}, {y.requires_grad=:}, {y._grad_fn=:}, {y.grad}')
print(f'{z=:}, {z.requires_grad=:}, {z._grad_fn=:}, {z.grad}')
