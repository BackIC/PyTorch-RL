'''import torch.nn.functional as F

from torch.distributions.categorical import Categorical
import torch


a = torch.tensor([[0.4985, 0.5015],
        [0.4923, 0.5077],
        [0.4979, 0.5021],
        [0.5029, 0.4971],
        [0.5094, 0.4906],
        [0.5102, 0.4898],
        [0.5097, 0.4903],
        [0.5033, 0.4967],
        [0.5098, 0.4902],
        [0.5036, 0.4964]])
m = Categorical(a)
print(m.log_prob(a))
'''

# import torch
# a = torch.Tensor([0.999, 0.001])
# b = a.gather(-1, torch.LongTensor([0, 1]))
# print(b) # 0.4985 / 0.5015
# c = b.log()
# print(c) # -0.6962 / -0.6902

# action_probs : tensor([0.4985, 0.5015], grad_fn=<SoftmaxBackward>)
# dist.probs : tensor([-0.6961, -0.6902], grad_fn=<SqueezeBackward1>)
# dist.sample : tensor(0)