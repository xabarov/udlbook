# newton.py
import numpy as np

class Newton:
    def __init__(self, x, damping=1e-4, damping_growth=10.0, max_tries=10):
        """
        x: np.ndarray параметров (вектор)
        damping: начальная лямбда (Tikhonov/LM демпфирование)
        """
        self.x = x
        self.damping = damping
        self.damping_growth = damping_growth
        self.max_tries = max_tries

    def step(self, grad, hess, loss=None, loss_fn=None):
        """
        grad: g(x) shape (P,)
        hess: H(x) shape (P,P)
        loss_fn: callable, если задан - делаем "проверку шага" (уменьшилась ли loss)
        """
        P = self.x.size
        I = np.eye(P, dtype=self.x.dtype)

        lam = self.damping
        x0 = self.x.copy()
        loss0 = loss if loss is not None else (loss_fn(x0) if loss_fn is not None else None)

        for _ in range(self.max_tries):
            H_lm = hess + lam * I
            try:
                delta = np.linalg.solve(H_lm, grad)
            except np.linalg.LinAlgError:
                lam *= self.damping_growth
                continue

            x_new = x0 - delta

            if loss_fn is None:
                self.x[:] = x_new
                self.damping = lam
                return

            loss_new = loss_fn(x_new)
            if loss0 is None or loss_new <= loss0:
                self.x[:] = x_new
                self.damping = max(lam / self.damping_growth, 1e-12)
                return
            else:
                lam *= self.damping_growth

        # если не нашли хороший шаг — делаем очень маленький "градиентный" шаг
        self.x[:] = x0 - 1e-6 * grad
        self.damping = lam