import numpy as np
import torch
x = torch.tensor([3,3])
print(x.shape)
y = x.dot(x)

z = torch.mul(x,x) #x.mul(x)

print(y)
print(z)