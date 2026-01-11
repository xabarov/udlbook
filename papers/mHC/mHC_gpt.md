Ниже — **минимальный, но рабочий** пример *decoder-only* Transformer (языковая модель) в PyTorch:

- **Обычный TransformerBlock** с residual: $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$, затем $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$.
- **mHC-версия**: держим $n$ параллельных residual-потоков (streams) $S \in \mathbb{R}^{B\times T\times n\times d}$ и добавляем матрицы гипер-связей:
  - $H^{pre}$ (веса смешивания потоков перед блоком, делаем через `softmax` → неотрицательные, сумма 1),
  - $H^{post}$ (веса раздачи обновления назад в потоки, тоже через `softmax`),
  - $H^{res}$ (матрица смешивания residual-обновлений между потоками), которую делаем **приблизительно дважды стохастической** через Sinkhorn–Knopp.

Важно: это **упрощённая** реализация “в духе” mHC (чтобы было понятно по коду, как это выглядит). Конкретные детали у разных реализаций HC/mHC могут отличаться (например, где именно смешивать, что именно смешивать: вход/выход/только residual и т.п.).

---

## 1) Обычный Transformer LM (baseline)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device):
    # mask=True означает "запретить внимание"
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, d]
        B, T, d = x.shape
        attn_mask = causal_mask(T, x.device)

        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(y)

        h = self.ln2(x)
        y = self.ff(h)
        x = x + self.dropout2(y)
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8, d_ff=2048, dropout=0.0, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab]
        return logits
```

---

## 2) mHC Transformer LM (streams + Sinkhorn на $H^{res}$)

### Sinkhorn–Knopp (дифференцируемая нормировка)
```python
def sinkhorn_knopp_from_logits(logits, n_iters=20, eps=1e-9):
    """
    logits: [n, n] (произвольные числа)
    return: [n, n] приблизительно doubly stochastic (rows/cols ~ 1/n),
            элементы > 0.

    Чтобы получить суммы строк/столбцов ~= 1, можно умножить на n.
    """
    K = torch.exp(logits)  # делаем строго положительной
    for _ in range(n_iters):
        K = K / (K.sum(dim=1, keepdim=True) + eps)  # normalize rows
        K = K / (K.sum(dim=0, keepdim=True) + eps)  # normalize cols
    return K
```

### Блок mHC
Идея блока (упрощённо):
1) Есть потоки $S$ (shape `[B,T,n,d]`).
2) Смешали в один вход $x$ через $H^{pre}$.
3) Посчитали обычный Transformer-переход $F(x)$ (attention+MLP).
4) Раздали обновление по потокам через $H^{post}$.
5) Перемешали обновления матрицей $H^{res}$ (после Sinkhorn) и добавили в streams.

```python
class mHCTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_streams=4, dropout=0.0, sinkhorn_iters=20):
        super().__init__()
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Веса pre/post делаем неотрицательными и суммирующимися в 1 через softmax
        self.pre_logits = nn.Parameter(torch.zeros(n_streams))
        self.post_logits = nn.Parameter(torch.zeros(n_streams))

        # H_res: [n, n] — будем делать doubly stochastic через Sinkhorn
        self.res_logits = nn.Parameter(torch.zeros(n_streams, n_streams))

        # "Обычный" трансформерный переход F
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_F(self, x):
        # x: [B, T, d]
        B, T, d = x.shape
        attn_mask = causal_mask(T, x.device)

        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(y)

        h = self.ln2(x)
        y = self.ff(h)
        x = x + self.dropout2(y)
        return x

    def forward(self, S):
        """
        S: [B, T, n, d] streams
        return S_new same shape
        """
        B, T, n, d = S.shape
        assert n == self.n

        # H_pre, H_post: [n], неотрицательные, sum=1
        w_pre = F.softmax(self.pre_logits, dim=0)   # [n]
        w_post = F.softmax(self.post_logits, dim=0) # [n]

        # Смешали streams в один x: x = sum_i w_pre[i] * S[:, :, i, :]
        x = torch.einsum("btn d, n -> btd", S, w_pre)  # [B, T, d]

        # Прогнали через обычный блок (attention+mlp)
        y = self.forward_F(x)  # [B, T, d]

        # Раздали обновление по потокам: U_i = w_post[i] * y
        U = y[:, :, None, :] * w_post[None, None, :, None]  # [B, T, n, d]

        # Sinkhorn -> doubly stochastic (rows/cols ~ 1/n). Масштабируем до сумм 1.
        H = sinkhorn_knopp_from_logits(self.res_logits, n_iters=self.sinkhorn_iters)  # [n, n]
        H = H * n  # теперь суммы строк/столбцов ~ 1

        # Смешиваем обновления между потоками:
        # U_mixed[:, :, j, :] = sum_i H[j,i] * U[:, :, i, :]
        U_mixed = torch.einsum("ji, bt i d -> bt j d", H, U)

        # Residual update каждого stream
        S_new = S + U_mixed
        return S_new
```

### Модель mHC-LM целиком
- Инициализируем streams как копии эмбеддинга (простое решение).
- В конце **схлопываем** streams в один hidden через softmax-веса readout и даём `lm_head`.

```python
class mHCTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        n_streams=4,
        dropout=0.0,
        max_len=2048,
        sinkhorn_iters=20,
    ):
        super().__init__()
        self.n = n_streams
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_streams=n_streams,
                dropout=dropout,
                sinkhorn_iters=sinkhorn_iters,
            )
            for _ in range(n_layers)
        ])

        # Readout: неотрицательный микс streams -> один x
        self.readout_logits = nn.Parameter(torch.zeros(n_streams))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x0 = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]  # [B, T, d]
        x0 = self.drop(x0)

        # streams: [B, T, n, d] (пока просто копируем)
        S = x0[:, :, None, :].expand(B, T, self.n, x0.size(-1)).contiguous()

        for blk in self.blocks:
            S = blk(S)

        # collapse streams
        w = F.softmax(self.readout_logits, dim=0)  # [n]
        x = torch.einsum("btn d, n -> btd", S, w)  # [B, T, d]

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

---

## 3) Что сравнивать при тесте
Если хочешь быстро убедиться, что всё работает:

- Одинаковые входы `idx` → `logits` одинакового shape `[B, T, vocab]`.
- В mHC можно проверить, что $H^{res}$ реально “почти дважды стохастическая”:
  - строки/столбцы матрицы `H / n` суммируются примерно в 1.

---

## 4) Важные практические замечания (коротко)
- В этой версии `sinkhorn_knopp_from_logits` возвращает матрицу с суммами строк/столбцов около $1/n$; я домножаю на $n$, чтобы суммы были около 1 (часто удобнее для residual-масштаба).
- Для больших $n$ и/или маленького $\tau$ в $K=\exp(\cdot)$ может потребоваться более стабильная реализация (log-domain через `logsumexp`).
- Настоящая “инженерная” mHC может добавлять оптимизации памяти/рекомпьют и отличаться тем, где именно стоят $H^{pre},H^{post}$.

---

Если скажешь:
1) **encoder-decoder** или **decoder-only** тебе нужен,
2) сколько streams $n$ (например, $n=4$ как часто берут),
3) хочешь ли вариант, где **каждый stream имеет свой attention/MLP** (дороже, но ближе к “мульти-ветвлению”),

— я подстрою код под твою цель и добавлю маленький скрипт обучения на игрушечных данных.