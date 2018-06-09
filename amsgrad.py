import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def AMSGrad(cost, params, lr=0.0001, b1=0.9, b2=0.999, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)

    for p, g in zip(params, grads):
        m = theano.shared(np.float32(p.get_value() * 0.))
        v = theano.shared(np.float32(p.get_value() * 0.))
        v_hat = theano.shared(np.float32(p.get_value() * 0.))

        m_t = np.float32(1 - b1) * m + np.float32(1. - b1) * g
        v_t = np.float32(1 - b2) * v + np.float32(1. - b2) * g ** 2
        v_hat_t = T.maximum(v_hat, v_t)
        p_t = p - lr_t * m_t / (T.sqrt(v_hat_t) + e)


        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((v_hat, v_hat_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    return updates
