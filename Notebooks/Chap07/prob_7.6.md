Возьмём конкретный пример, чтобы «глазами» увидеть, почему получается именно $\Omega^T$ при вашей договорённости о том, где стоит $\frac{\partial z_i}{\partial h_j}$.

---

## Пример

Пусть
- $h \in \mathbb{R}^3$, то есть $h=\begin{pmatrix}h_1\\h_2\\h_3\end{pmatrix}$,
- $z \in \mathbb{R}^2$, то есть $z=\begin{pmatrix}z_1\\z_2\end{pmatrix}$,
- $\beta=\begin{pmatrix}\beta_1\\\beta_2\end{pmatrix}$,
- $\Omega \in \mathbb{R}^{2\times 3}$, например
$$
\Omega=
\begin{pmatrix}
\omega_{11} & \omega_{12} & \omega_{13}\\
\omega_{21} & \omega_{22} & \omega_{23}
\end{pmatrix}.
$$

И задано:
$$
z=\beta+\Omega h.
$$

---

## 1) Расписываем $z$ по компонентам

Перемножим $\Omega h$:

$$
\Omega h=
\begin{pmatrix}
\omega_{11} & \omega_{12} & \omega_{13}\\
\omega_{21} & \omega_{22} & \omega_{23}
\end{pmatrix}
\begin{pmatrix}
h_1\\h_2\\h_3
\end{pmatrix}
=
\begin{pmatrix}
\omega_{11}h_1+\omega_{12}h_2+\omega_{13}h_3\\
\omega_{21}h_1+\omega_{22}h_2+\omega_{23}h_3
\end{pmatrix}.
$$

Тогда
$$
z=
\begin{pmatrix}
z_1\\z_2
\end{pmatrix}
=
\begin{pmatrix}
\beta_1\\\beta_2
\end{pmatrix}
+
\begin{pmatrix}
\omega_{11}h_1+\omega_{12}h_2+\omega_{13}h_3\\
\omega_{21}h_1+\omega_{22}h_2+\omega_{23}h_3
\end{pmatrix}.
$$

То есть явно:
$$
z_1=\beta_1+\omega_{11}h_1+\omega_{12}h_2+\omega_{13}h_3,
$$
$$
z_2=\beta_2+\omega_{21}h_1+\omega_{22}h_2+\omega_{23}h_3.
$$

---

## 2) Находим все $\frac{\partial z_i}{\partial h_j}$

Теперь просто дифференцируем:

### Для $z_1$
$$
\frac{\partial z_1}{\partial h_1}=\omega_{11},\quad
\frac{\partial z_1}{\partial h_2}=\omega_{12},\quad
\frac{\partial z_1}{\partial h_3}=\omega_{13}.
$$

### Для $z_2$
$$
\frac{\partial z_2}{\partial h_1}=\omega_{21},\quad
\frac{\partial z_2}{\partial h_2}=\omega_{22},\quad
\frac{\partial z_2}{\partial h_3}=\omega_{23}.
$$

То есть в общем виде действительно $\frac{\partial z_i}{\partial h_j}=\omega_{ij}$.

---

## 3) Собираем матрицу $\frac{\partial z}{\partial h}$ по вашей конвенции

Вы написали:

> $\frac{\partial z}{\partial h}$ — матрица, содержащая $\frac{\partial z_i}{\partial h_j}$ в **$i$-м столбце** и **$j$-й строке**.

Значит:
- **строки** соответствуют $h_j$ (то есть $j=1,2,3$),
- **столбцы** соответствуют $z_i$ (то есть $i=1,2$).

И матрица будет размера $3\times 2$:

$$
\frac{\partial z}{\partial h}
=
\begin{pmatrix}
\frac{\partial z_1}{\partial h_1} & \frac{\partial z_2}{\partial h_1}\\
\frac{\partial z_1}{\partial h_2} & \frac{\partial z_2}{\partial h_2}\\
\frac{\partial z_1}{\partial h_3} & \frac{\partial z_2}{\partial h_3}
\end{pmatrix}.
$$

Подставляем найденные производные:

$$
\frac{\partial z}{\partial h}
=
\begin{pmatrix}
\omega_{11} & \omega_{21}\\
\omega_{12} & \omega_{22}\\
\omega_{13} & \omega_{23}
\end{pmatrix}.
$$

---

## 4) Сравниваем с $\Omega^T$

Транспонируем $\Omega$:

$$
\Omega^T=
\begin{pmatrix}
\omega_{11} & \omega_{21}\\
\omega_{12} & \omega_{22}\\
\omega_{13} & \omega_{23}
\end{pmatrix}.
$$

Видно, что
$$
\frac{\partial z}{\partial h}=\Omega^T.
$$

---

<details>
<summary>Небольшое пояснение, почему не просто $\Omega$</summary>

Часто Якобиан определяют как матрицу $J$, где элемент $J_{ij}=\frac{\partial z_i}{\partial h_j}$ (то есть $i$ — строка, $j$ — столбец). Тогда получилось бы $J=\Omega$.

Но у вас **наоборот**: $\frac{\partial z_i}{\partial h_j}$ стоит в $j$-й строке и $i$-м столбце, поэтому матрица получается транспонированной: $\Omega^T$.
</details>

---

Если хотите, могу сделать тот же пример с конкретными числами (например, $\Omega=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$) и прямо выписать все частные производные в таблице.