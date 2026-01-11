Focal Loss — это модификация кросс-энтропии, которая **уменьшает вклад “лёгких” примеров** и фокусирует обучение на **трудных/редких**. Чаще всего её используют при **сильном дисбалансе классов** или когда модель быстро “забивает” на сложные примеры.

## Идея и формула

### Бинарная классификация
Пусть $y \in \{0,1\}$, $p$ — предсказанная вероятность класса $1$.

Определим
- $p_t = p$ если $y=1$, иначе $p_t = 1-p$
- $\alpha_t = \alpha$ если $y=1$, иначе $\alpha_t = 1-\alpha$

Тогда **Focal Loss**:
$$
\mathrm{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- $\gamma \ge 0$ — **focusing parameter**: чем больше, тем сильнее “давим” лёгкие примеры.
- $\alpha \in [0,1]$ — **балансировка классов** (особенно полезна при дисбалансе).

При $\gamma=0$ получается обычная взвешенная BCE.

---

## Мини-пример: чем “лёгкий” отличается от “трудного”

Ниже покажем, что для “лёгкого” примера (высокий $p_t$) множитель $(1-p_t)^\gamma$ резко уменьшает лосс.

```python
import torch

def focal_loss_binary_probs(p, y, alpha=0.25, gamma=2.0, eps=1e-8, reduction="none"):
    """
    p: probabilities for class 1, shape [N]
    y: labels in {0,1}, shape [N]
    """
    p = torch.clamp(p, eps, 1 - eps)
    y = y.float()

    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.tensor(alpha, device=p.device), torch.tensor(1 - alpha, device=p.device))

    loss = -at * (1 - pt) ** gamma * torch.log(pt)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

# "лёгкий" позитивный: p=0.99 при y=1
# "трудный" позитивный: p=0.10 при y=1
p = torch.tensor([0.99, 0.10])
y = torch.tensor([1, 1])

print("per-sample FL:", focal_loss_binary_probs(p, y, alpha=0.25, gamma=2.0))
print("per-sample FL (gamma=0):", focal_loss_binary_probs(p, y, alpha=0.25, gamma=0.0))
```

Ожидаемо:
- при $\gamma=2$ лосс для $p=0.99$ станет **почти нулевым**,
- а для $p=0.10$ останется **большим**.

---

## Практичная реализация: бинарный focal loss по logits (стабильно)

На практике модель выдаёт **логиты** $z$, а $p=\sigma(z)$. Важно считать устойчиво (через log-sigmoid), чтобы не ловить NaN.

```python
import torch
import torch.nn.functional as F

def focal_loss_binary_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    logits: shape [N] or [N, ...]
    targets: same shape, in {0,1}
    """
    targets = targets.float()

    # log p and log (1-p) in stable form
    log_p = F.logsigmoid(logits)         # log(sigmoid(z))
    log_1mp = F.logsigmoid(-logits)      # log(1 - sigmoid(z))

    # p and (1-p) if нужны для modulating factor
    p = torch.sigmoid(logits)

    # p_t and log(p_t)
    pt = torch.where(targets == 1, p, 1 - p)
    log_pt = torch.where(targets == 1, log_p, log_1mp)

    # alpha_t
    at = torch.where(targets == 1, torch.tensor(alpha, device=logits.device), torch.tensor(1 - alpha, device=logits.device))

    loss = -at * (1 - pt) ** gamma * log_pt

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
```

### Использование в простом цикле обучения (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# toy model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1)  # binary logit
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# fake batch
x = torch.randn(128, 20)
y = (torch.rand(128) < 0.05).float()  # 5% positives (дисбаланс)
logits = model(x).squeeze(-1)

loss = focal_loss_binary_with_logits(logits, y, alpha=0.75, gamma=2.0)  # alpha можно ↑ для редкого класса
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("loss:", float(loss))
```

---

## Мультикласс: focal loss по logits (softmax)

Для $K$ классов, логиты $z \in \mathbb{R}^K$, $p=\mathrm{softmax}(z)$, целевой класс $y$.

$$
\mathrm{FL} = -\alpha_y (1 - p_y)^\gamma \log(p_y)
$$

Реализация:

```python
import torch
import torch.nn.functional as F

def focal_loss_multiclass_with_logits(logits, targets, alpha=None, gamma=2.0, reduction="mean"):
    """
    logits: shape [N, C]
    targets: shape [N] with class indices in [0..C-1]
    alpha: None or tensor/list of shape [C] with class weights
    """
    N, C = logits.shape
    log_probs = F.log_softmax(logits, dim=1)   # log p_c
    probs = log_probs.exp()                   # p_c

    # pick p_y and log(p_y)
    targets = targets.long()
    idx = torch.arange(N, device=logits.device)
    log_pt = log_probs[idx, targets]
    pt = probs[idx, targets]

    if alpha is None:
        at = torch.ones_like(pt)
    else:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
        at = alpha[targets]

    loss = -at * (1 - pt) ** gamma * log_pt

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

# quick demo
logits = torch.randn(8, 4)         # 8 samples, 4 classes
targets = torch.randint(0, 4, (8,))
alpha = [1.0, 2.0, 1.0, 3.0]       # например, усилить редкие классы 1 и 3
print(focal_loss_multiclass_with_logits(logits, targets, alpha=alpha, gamma=2.0))
```

---

## Как выбирать $\alpha$ и $\gamma$

- Обычно начинают с $\gamma=2$.
- Для бинарного детекта (как в оригинальной статье RetinaNet) часто берут $\alpha=0.25$ для позитивов, но это не универсально.
- При сильном дисбалансе можно:
  - поставить $\alpha$ больше для редкого класса,
  - или использовать $\alpha$ как вектор весов классов в мультиклассе.

---

<details>
<summary><b>Практические замечания и типичные грабли</b></summary>

- Focal loss может **ухудшать калибровку вероятностей**: модель сильнее оптимизирует “сложные” примеры, а не правдоподобные вероятности.
- Если $\gamma$ слишком большой, градиенты по “лёгким” примерам почти исчезают → обучение может стать нестабильным или медленным.
- Часто имеет смысл сравнить с простыми базовыми вариантами:
  - Weighted BCE / Weighted CE
  - Oversampling / undersampling
  - Class-balanced loss
- Для segmentation обычно считают лосс по всем пикселям и затем делают `mean` по пикселям и батчу (или аккуратно маскируют ignore-index).

</details>

Если скажешь, какой у тебя кейс (бинарный/мультикласс/сегментация, насколько сильный дисбаланс, какие метрики важны), я предложу конкретные настройки $\alpha,\gamma$ и корректный способ усреднения лосса.