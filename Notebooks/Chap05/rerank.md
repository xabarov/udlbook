Для задач ранжирования “чаще всего” нет одного универсального лосса — используют **pointwise / pairwise / listwise** подходы. Для **BERT-реранкеров (обычно cross-encoder: [query; doc] → score)** на практике чаще всего встречаются:

1) **Pointwise classification/regression** (кросс-энтропия или MSE по метке релевантности) — самый распространённый baseline.  
2) **Pairwise** (RankNet logistic / hinge / margin ranking) — очень популярно, когда в данных “пары” $(d^+, d^-)$ на один запрос.  
3) **Listwise softmax по кандидатам на один запрос** (softmax cross-entropy по группе кандидатов; часто с in-batch negatives) — также очень распространено в современных пайплайнах, потому что хорошо сочетается с “1 позитив + много негативов” и напрямую оптимизирует относительный порядок.

Ниже — что именно обычно делают **для BERT-reranker** и как это выглядит в коде.

---

## 1) Pointwise (самый частый baseline для cross-encoder reranker)

Модель выдаёт скаляр $s(q,d)$. Дальше:

- если метки бинарные $y \in \{0,1\}$: **BCEWithLogitsLoss**  
$$
\mathcal{L} = -y\log \sigma(s) - (1-y)\log(1-\sigma(s))
$$

- если метки градуированные (0/1/2/3…): либо **CrossEntropy** по классам релевантности, либо регрессия (MSE), но CE чаще.

```python
import torch
import torch.nn.functional as F

# scores: [N] logits, y: [N] in {0,1}
scores = torch.randn(32)
y = torch.randint(0, 2, (32,)).float()

loss = F.binary_cross_entropy_with_logits(scores, y)
```

Когда это выбирают:
- когда есть “абсолютные” метки релевантности,
- когда хочется простой и стабильной оптимизации,
- когда обучают на независимых примерах (без группировки по запросу).

Минус: ранжирование — относительная задача, а pointwise не видит “конкурентов” внутри одного запроса.

---

## 2) Pairwise (очень типично для ранжирования)

Идея: для каждого запроса хотим $s^+ > s^-$.  

### RankNet (pairwise logistic)
$$
\mathcal{L} = \log\left(1 + \exp(-(s^+ - s^-))\right)
$$

```python
import torch
import torch.nn.functional as F

s_pos = torch.randn(64)  # scores for relevant docs
s_neg = torch.randn(64)  # scores for non-relevant docs

loss = F.softplus(-(s_pos - s_neg)).mean()
```

### Hinge / margin ranking
$$
\mathcal{L} = \max(0, m - (s^+ - s^-))
$$

```python
m = 1.0
loss = torch.relu(m - (s_pos - s_neg)).mean()
```

Когда это выбирают:
- когда датасет естественно даёт пары $(q, d^+, d^-)$ (например, клики/логика формирования негативов),
- когда важен относительный порядок больше, чем калиброванная вероятность.

---

## 3) Listwise softmax по кандидатам (часто “де-факто стандарт” в reranking с $1$ позитивом и $K$ негативами)

Для каждого запроса берём набор кандидатов $\{d_0,\dots,d_K\}$, среди них один позитив (или несколько). Модель выдаёт scores $s_i$. Если позитив один и его индекс $i^*$, то лосс:

$$
\mathcal{L} = -\log \frac{\exp(s_{i^*})}{\sum_j \exp(s_j)}
$$

Это по сути **CrossEntropy по группе кандидатов** (очень похоже на InfoNCE).

```python
import torch
import torch.nn.functional as F

# scores per query: [B, K] (B queries, K candidates each)
B, K = 8, 16
scores = torch.randn(B, K)

# target index of the positive doc in each row
pos_idx = torch.randint(0, K, (B,))

loss = F.cross_entropy(scores, pos_idx)
```

Когда это выбирают:
- MS MARCO-подобные режимы (1 позитив + много негативов),
- in-batch negatives,
- когда хочется, чтобы модель училась “выигрывать” у конкурентов в рамках запроса.

---

## Что “чаще” именно для BERT reranker?

Если говорить про типичный **BERT cross-encoder reranker** в статьях/репозиториях:

- Очень часто начинают с **pointwise BCE/CE** (простота).
- Очень часто в сильных системах используют **listwise softmax по группе кандидатов** (особенно когда кандидаты формируются retriever’ом и есть набор на запрос).
- **Pairwise** тоже широко используется, особенно когда удобно генерировать пары и хочется прямого давления на разницу $s^+-s^-$.

На практике “победитель” зависит от того, **как у тебя устроены данные**:
- есть ли группировка по запросу,
- сколько позитивов на запрос,
- как добываются негативы (случайные / hard negatives),
- какая метрика целевая (MRR, nDCG@k).

---

<details>
<summary><b>Быстрый выбор: что взять в качестве первого рабочего варианта</b></summary>

- Если у тебя для каждого запроса есть список кандидатов (retrieval → rerank) и в списке известен(ы) релевантный(е): начни с **listwise softmax CE** по группе кандидатов.
- Если данные в формате “позитив/негатив” пар: **RankNet logistic** (softplus) обычно хороший и гладкий выбор.
- Если только независимые (q,d) с меткой релевантности: **pointwise BCE/CE**.

</details>

Если напишешь, в каком формате у тебя батчи (по одному $(q,d)$ или группы кандидатов на запрос), сколько позитивов на запрос и какая метрика (MRR@10, nDCG@10 и т.п.), подскажу конкретный лосс и схему семплирования негативов (это часто важнее выбора между RankNet vs listwise CE).