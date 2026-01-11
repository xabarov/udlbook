import numpy as np
from optimizers import SGD, SGDMomentum, SGDNesterov

# synthetic data
rng = np.random.default_rng(0)
N = 256
x = rng.normal(size=(N, 1))
w_true = np.array([[2.0]])
b_true = np.array([[-1.0]])
y = x @ w_true + b_true + 0.1 * rng.normal(size=(N, 1))

# parameters
w = rng.normal(size=(1, 1))
b = np.zeros((1, 1))
params = [w, b]


def loss_and_grads(x, y, w, b):
    pred = x @ w + b
    err = pred - y
    loss = (err**2).mean()

    # d/dpred: 2/N * err
    dp = (2.0 / x.shape[0]) * err
    dw = x.T @ dp
    db = dp.sum(axis=0, keepdims=True)
    return loss, [dw, db]


# pick one optimizer:
opt = SGDNesterov(params, lr=0.1, momentum=0.9)
# opt = SGDMomentum(params, lr=0.1, momentum=0.9)
# opt = SGD(params, lr=0.1)

for t in range(200):
    loss, grads = loss_and_grads(x, y, w, b)
    opt.step(grads)
    if (t + 1) % 50 == 0:
        print(
            f"iter {t+1:4d} | loss={loss:.6f} | w={w.ravel()[0]:.3f} b={b.ravel()[0]:.3f}"
        )
