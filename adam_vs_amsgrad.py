import numpy as np
import theano
import theano.tensor as T
from theano import function
from adam import Adam
from amsgrad import AMSGrad
import matplotlib.pyplot as plt


def optimize(optimizer, num_iteration=int(1e+6)):
    x = theano.shared(np.float32(0.))

    rs = np.random.RandomState(1234)
    rng = T.shared_randomstreams.RandomStreams(rs.randint(999999))
    r = rng.binomial(n=1, p=(0.01), size=(1,))

    cost = 1010 * T.clip(x, -1, 1) * r[0] + -10 * T.clip(x, -1, 1) * (1-r[0])

    f = function([], cost, updates=optimizer(cost, (x,), lr=1e-4, b1=0.9, b2=0.999, e=1e-8))

    loss_history = np.empty((0))

    for i in range(num_iteration):
        loss_val = f()

        avg_loss_val = 1010 * x.get_value() * 0.01 + -10 * x.get_value() * 0.99

        loss_history = np.append(loss_history, avg_loss_val)
        print(avg_loss_val, end='\t\r')
    print('\n')

    return np.arange(len(loss_history)), loss_history


def main():
    t, loss = optimize(Adam)
    plt.plot(t, loss, color='r', label='Adam')
    t, loss = optimize(AMSGrad)
    plt.plot(t, loss, color='b', label='AMSGrad')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.savefig('adam_va_amsgrad.png')


if __name__ == '__main__':
    main()
