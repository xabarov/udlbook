We start from the usual **discriminative** linear regression model and then reformulate it as a **generative** model.

---

## 1. Discriminative formulation

The standard linear regression model is

$$
y = f(x;\,\phi) = \phi_0 + \phi_1 x
$$

with squared-error loss

$$
\mathcal{L}_{\text{disc}}(\phi)
= \sum_{i=1}^N \left(y_i - (\phi_0 + \phi_1 x_i)\right)^2
$$

This directly models $p(y \mid x)$.

---

## 2. Generative reformulation

Now we instead model **$x$ given $y$**:

$$
x = g(y;\,\phi) = \phi_0 + \phi_1 y
$$

This corresponds to modeling $p(x \mid y)$.

### New loss function

Using squared error again, the generative loss is

$$
\mathcal{L}_{\text{gen}}(\phi)
= \sum_{i=1}^N \left(x_i - (\phi_0 + \phi_1 y_i)\right)^2
$$

This is just linear regression with the roles of $x$ and $y$ swapped.

---

## 3. Inference: inverse function

To make predictions of $y$ given a new $x$, we must invert the generative model.

From

$$
x = \phi_0 + \phi_1 y
$$

solve for $y$:

$$
y = g^{-1}(x;\,\phi)
= \frac{x - \phi_0}{\phi_1}
\quad \text{(assuming } \phi_1 \neq 0\text{)}
$$

This inverse is what we use for inference after fitting the generative model.

---

## 4. Are the predictions the same?

**In general: no.**

### Why?

- The discriminative model minimizes **vertical errors** in $y$.
- The generative model minimizes **horizontal errors** in $x$.
- These correspond to different objective functions unless the data lie exactly on a straight line or have very special symmetry.

Only in special cases (e.g., noiseless data or perfectly symmetric noise) will both approaches produce identical predictions.

---

## 5. Numerical example with three points

Below is a simple Python example that fits both models to three data points and compares the resulting prediction functions.

```python
import numpy as np

# three data points
x = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 2.0])

# --- Discriminative: y = a + b x ---
X = np.vstack([np.ones_like(x), x]).T
phi_disc = np.linalg.lstsq(X, y, rcond=None)[0]

# --- Generative: x = c + d y ---
Y = np.vstack([np.ones_like(y), y]).T
phi_gen = np.linalg.lstsq(Y, x, rcond=None)[0]

# Invert generative model: y = (x - c) / d
def y_from_gen(x, phi):
    return (x - phi[0]) / phi[1]

print("Discriminative parameters (phi0, phi1):", phi_disc)
print("Generative parameters (phi0, phi1):", phi_gen)

# Compare predictions
x_test = np.array([1.5, 2.5])
y_disc_pred = phi_disc[0] + phi_disc[1] * x_test
y_gen_pred = y_from_gen(x_test, phi_gen)

print("Discriminative predictions:", y_disc_pred)
print("Generative predictions:", y_gen_pred)
```

### Typical outcome

- The fitted parameters differ.
- The predicted $y$ values differ.
- Hence, the generative and discriminative models do **not** generally agree.

---

## 6. Summary

- **New loss function (generative):**
  $$
  \sum_i \left(x_i - (\phi_0 + \phi_1 y_i)\right)^2
  $$

- **Inference rule:**
  $$
  y = \frac{x - \phi_0}{\phi_1}
  $$

- **Equivalence to discriminative regression?**
  - ❌ No, not in general.
  - ✅ Only in special/noiseless cases.

This highlights a key conceptual difference: **what you choose to model (cause vs. effect) matters for prediction**.