Рассмотрим стандартную линейную регрессию с двумя параметрами $\phi_0,\phi_1$ и функцией потерь MSE (квадратичная ошибка) на выборке $\{(x_i,y_i)\}_{i=1}^m$:
$$
J(\phi_0,\phi_1)=\frac{1}{2m}\sum_{i=1}^m\big(\phi_0+\phi_1 x_i-y_i\big)^2.
$$

## 1) Алгебраическое выражение для Гессиана $H[J]$

Обозначим ошибку на $i$-м объекте:
$$
e_i(\phi_0,\phi_1)=\phi_0+\phi_1 x_i-y_i.
$$
Тогда
$$
J=\frac{1}{2m}\sum_{i=1}^m e_i^2.
$$

Вычислим вторые производные.

**Первые производные:**
$$
\frac{\partial J}{\partial \phi_0}=\frac{1}{m}\sum_{i=1}^m e_i,\qquad
\frac{\partial J}{\partial \phi_1}=\frac{1}{m}\sum_{i=1}^m e_i x_i.
$$

**Вторые производные:**
- Так как $\frac{\partial e_i}{\partial \phi_0}=1$, получаем
$$
\frac{\partial^2 J}{\partial \phi_0^2}
=\frac{1}{m}\sum_{i=1}^m 1
=\frac{m}{m}=1.
$$
- Так как $\frac{\partial e_i}{\partial \phi_1}=x_i$,
$$
\frac{\partial^2 J}{\partial \phi_0\partial \phi_1}
=\frac{1}{m}\sum_{i=1}^m x_i,
\qquad
\frac{\partial^2 J}{\partial \phi_1^2}
=\frac{1}{m}\sum_{i=1}^m x_i^2.
$$

Итак, матрица Гессе (по параметрам $\phi_0,\phi_1$) равна
$$
H[J](\phi_0,\phi_1)=
\begin{pmatrix}
\frac{\partial^2 J}{\partial \phi_0^2} & \frac{\partial^2 J}{\partial \phi_0\partial \phi_1}\\
\frac{\partial^2 J}{\partial \phi_1\partial \phi_0} & \frac{\partial^2 J}{\partial \phi_1^2}
\end{pmatrix}
=
\frac{1}{m}
\begin{pmatrix}
m & \sum_{i=1}^m x_i\\
\sum_{i=1}^m x_i & \sum_{i=1}^m x_i^2
\end{pmatrix}.
$$

Заметьте, что этот Гессиан **не зависит от $\phi_0,\phi_1$**, то есть поверхность ошибки квадратична и имеет постоянную кривизну.

---

## 2) Доказательство выпуклости через след и определитель (собственные значения $>0$)

Для симметричной матрицы $2\times 2$
$$
H=\begin{pmatrix}a&b\\b&c\end{pmatrix}
$$
собственные значения $\lambda_1,\lambda_2$ удовлетворяют:
$$
\lambda_1+\lambda_2=\mathrm{tr}(H),\qquad \lambda_1\lambda_2=\det(H).
$$
Если $\mathrm{tr}(H)>0$ и $\det(H)>0$, то обе $\lambda_1,\lambda_2>0$ (а значит $H$ положительно определена, и функция строго выпукла).

### След Гессиана
Для нашего $H[J]$:
$$
\mathrm{tr}(H[J])=\frac{1}{m}\left(m+\sum_{i=1}^m x_i^2\right).
$$
Так как $m>0$ и $\sum x_i^2\ge 0$, имеем
$$
\mathrm{tr}(H[J])>0.
$$

### Определитель Гессиана
Посчитаем:
$$
\det(H[J])=\frac{1}{m^2}\left(m\sum_{i=1}^m x_i^2-\left(\sum_{i=1}^m x_i\right)^2\right).
$$
Выражение в скобках — это (с точностью до множителя) неотрицательная величина из неравенства Коши–Буняковского:
$$
\left(\sum_{i=1}^m x_i\right)^2 \le m\sum_{i=1}^m x_i^2
\;\Rightarrow\;
m\sum_{i=1}^m x_i^2-\left(\sum_{i=1}^m x_i\right)^2 \ge 0.
$$
Следовательно,
$$
\det(H[J])\ge 0.
$$

Более того, можно переписать через дисперсию:
$$
m\sum x_i^2-\left(\sum x_i\right)^2
= m^2\,\mathrm{Var}(x),
$$
где
$$
\mathrm{Var}(x)=\frac{1}{m}\sum_{i=1}^m (x_i-\bar x)^2,\qquad \bar x=\frac{1}{m}\sum_{i=1}^m x_i.
$$
Отсюда
$$
\det(H[J])=\mathrm{Var}(x)\ge 0,
$$
(с учётом того, что у нас в $H[J]$ уже стоит множитель $1/m$; в точности получается $\det(H[J])=\frac{1}{m^2}m^2\mathrm{Var}(x)=\mathrm{Var}(x)$).

---

## 3) Вывод: выпуклость и когда она строгая

- Поскольку $H[J]$ симметрична и $\det(H[J])\ge 0$, $\mathrm{tr}(H[J])>0$, то собственные значения **неотрицательны**: $\lambda_1,\lambda_2\ge 0$. Значит $J(\phi_0,\phi_1)$ **выпукла**.

- Чтобы собственные значения были **строго положительными** ($\lambda_1,\lambda_2>0$), нужно $\det(H[J])>0$, то есть
$$
\mathrm{Var}(x)>0,
$$
что эквивалентно тому, что **не все $x_i$ одинаковы**. В этом случае $H[J]$ положительно определена, $J$ **строго выпукла** и минимум единственный.

Если хотите, могу также вывести собственные значения явно формулой для $2\times 2$ и показать, что они выражаются через $\mathrm{tr}$ и $\det$ и потому положительны при $\det>0$.