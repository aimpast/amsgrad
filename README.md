# amsgrad

## Usage

The implementation of AMSGrad which is proposed by [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)(ICLR2018) in Theano.

You can carry out the comparison experiment between Adam and AMSGrad as following command:
```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python adam_vs_amsgrad.py
```


## References
[1] [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

[2] https://gist.github.com/Newmu/acb738767acb4788bac3
