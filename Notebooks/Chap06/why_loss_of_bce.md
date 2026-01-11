Ниже аккуратно распишу, почему при BCE + сигмоиде получается

$$
\frac{\partial \mathcal L}{\partial z}=\hat y-y,
\quad \text{где } \hat y=\sigma(z).
$$

---

## 1) Что задано

Логит:
$$
z=\phi_0+\phi_1 x.
$$

Сигмоида (вероятность):
$$
\hat y=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

BCE (для одного объекта):
$$
\mathcal L(\hat y,y)=-\Big(y\log \hat y+(1-y)\log(1-\hat y)\Big).
$$

Мы хотим найти $\frac{\partial \mathcal L}{\partial z}$.

---

## 2) Правило цепочки

$\mathcal L$ зависит от $z$ через $\hat y=\sigma(z)$, поэтому:

$$
\frac{\partial \mathcal L}{\partial z}
=
\frac{\partial \mathcal L}{\partial \hat y}\cdot \frac{\partial \hat y}{\partial z}.
$$

Дальше считаем оба множителя.

---

## 3) Считаем $\frac{\partial \mathcal L}{\partial \hat y}$

Возьмём производную BCE по $\hat y$:

$$
\mathcal L(\hat y,y)=-(y\log \hat y+(1-y)\log(1-\hat y)).
$$

Тогда

$$
\frac{\partial \mathcal L}{\partial \hat y}
=
-\left(
y\cdot \frac{1}{\hat y}
+
(1-y)\cdot \frac{\partial}{\partial \hat y}\log(1-\hat y)
\right).
$$

А
$$
\frac{\partial}{\partial \hat y}\log(1-\hat y)=\frac{-1}{1-\hat y}.
$$

Подставляем:

$$
\frac{\partial \mathcal L}{\partial \hat y}
=
-\left(
\frac{y}{\hat y}
+
(1-y)\cdot \frac{-1}{1-\hat y}
\right)
=
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}.
$$

Итак:

$$
\frac{\partial \mathcal L}{\partial \hat y}
=
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}.
$$

---

## 4) Считаем $\frac{\partial \hat y}{\partial z}$ для сигмоиды

$$
\hat y=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Известная производная сигмоиды:

$$
\frac{\partial \hat y}{\partial z}=\hat y(1-\hat y).
$$

(Если нужно, могу отдельно вывести это тоже.)

---

## 5) Перемножаем и упрощаем до $\hat y-y$

Теперь:

$$
\frac{\partial \mathcal L}{\partial z}
=
\left(
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}
\right)\cdot \hat y(1-\hat y).
$$

Раскроем произведение по слагаемым:

Первое слагаемое:
$$
-\frac{y}{\hat y}\cdot \hat y(1-\hat y)= -y(1-\hat y).
$$

Второе слагаемое:
$$
\frac{1-y}{1-\hat y}\cdot \hat y(1-\hat y)=(1-y)\hat y.
$$

Складываем:

$$
\frac{\partial \mathcal L}{\partial z}
=
-y(1-\hat y)+(1-y)\hat y.
$$

Упростим:

$$
-y(1-\hat y) = -y + y\hat y,
\qquad
(1-y)\hat y=\hat y-y\hat y.
$$

Тогда

$$
\frac{\partial \mathcal L}{\partial z}
=
(-y+y\hat y)+(\hat y-y\hat y)
=
-y+\hat y
=
\hat y-y.
$$

**Готово:**
$$
\frac{\partial \mathcal L}{\partial z}=\hat y-y.
$$

---

<details>
<summary>Альтернативный короткий вывод через производные логарифмов от сигмоиды</summary>

Запишем:

$$
\mathcal L(z)= -\Big(y\log \sigma(z)+(1-y)\log(1-\sigma(z))\Big).
$$

Нужны две производные:

1) 
$$
\frac{d}{dz}\log \sigma(z)=1-\sigma(z)=1-\hat y.
$$

2)
$$
\frac{d}{dz}\log(1-\sigma(z))=-\sigma(z)=-\hat y.
$$

Тогда:

$$
\frac{d\mathcal L}{dz}
=
-\Big(y(1-\hat y)+(1-y)(-\hat y)\Big)
=
-\Big(y-y\hat y-\hat y+y\hat y\Big)
=
\hat y-y.
$$
</details>

Если хочешь, дальше распишу переход от $\frac{\partial \mathcal L}{\partial z}$ к градиентам по $\phi_0,\phi_1$ (там просто домножение на $\frac{\partial z}{\partial \phi}$).