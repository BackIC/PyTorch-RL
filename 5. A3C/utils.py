"""
Functions that use multiple times
"""

from torch import nn
import torch
import numpy as np


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.1)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        # s_					[ 0.06607786  0.74773231 -0.10495857 -1.32181293]
        # 버퍼 이후의 다음상태

        # [None, :]		        [[ 0.06607786  0.74773231 -0.10495857 -1.32181293]]
        # tensor에 넣어주기 위해 한번더 둘러쌈
        # 안할시 IndexError: too many indices for array 발생

        # v_wrap			    tensor([[ 0.0661,  0.7477, -0.1050, -1.3218]])
        # 상태를 tensor 상태로 변경

        # forward		    	(tensor([[0.0637, 0.0626]], grad_fn=<ThAddmmBackward>), tensor([[0.2033]], grad_fn=<ThAddmmBackward>))
        # localNet에 forwarding, logits와 value 가져옴

        # -1					tensor([[0.2033]], grad_fn=<ThAddmmBackward>)
        # value만 추려냄

        # data			    	tensor([[0.2033]])
        # backpropagation 제거

        # numpy			        [[ 0.20328102]]
        # torch값을 numpy로 변환

        # [0, 0]				0.203281
        # 값만 추려옴




    #print('v_s_ :', v_s_)
    #print('gamma :', gamma)
    #print('br', br)
    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()
    #print('buffer_v_target :', buffer_v_target)

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )