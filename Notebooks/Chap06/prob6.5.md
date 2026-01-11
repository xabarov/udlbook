Пусть есть обучающая выборка $\{(x_n,y_n)\}_{n=1}^N$ и модель
$$
f(x;\phi,\theta)=\phi_0+\phi_1\,a(z_1)+\phi_2\,a(z_2)+\phi_3\,a(z_3),
$$
где
$$
z_1=\theta_{10}+\theta_{11}x,\quad
z_2=\theta_{20}+\theta_{21}x,\quad
z_3=\theta_{30}+\theta_{31}x,
$$
а $a(\cdot)$ — ReLU:
$$
a(z)=\max(0,z).
$$

Возьмём least squares loss в стандартной форме
$$
\mathcal L(\phi,\theta)=\frac12\sum_{n=1}^N\big(f(x_n;\phi,\theta)-y_n\big)^2.
$$
Обозначим остаток (ошибку) на объекте $n$:
$$
r_n = f(x_n;\phi,\theta)-y_n.
$$
Тогда по правилу цепочки
$$
\frac{\partial \mathcal L}{\partial p}=\sum_{n=1}^N r_n\,\frac{\partial f(x_n;\phi,\theta)}{\partial p}.
$$

## Производная ReLU (важно)
$$
a'(z)=\begin{cases}
1,& z>0,\\
0,& z<0,
\end{cases}
$$
а в точке $z=0$ производная не определена; обычно берут субградиент, например $a'(0)=0$ (часто) или $a'(0)=1$ — главное выбрать согласованно.

Удобно ввести индикатор “активности” нейрона:
$$
g_{i,n}=a'(z_i(x_n))=\mathbf 1[z_i(x_n)>0]\quad (\text{и }g_{i,n}=0\text{ при }z_i=0\text{ по соглашению}).
$$

---

## Градиенты по $\phi$ (4 параметра)
Для каждого $n$:
$$
\frac{\partial f_n}{\partial \phi_0}=1,\quad
\frac{\partial f_n}{\partial \phi_1}=a(z_{1,n}),\quad
\frac{\partial f_n}{\partial \phi_2}=a(z_{2,n}),\quad
\frac{\partial f_n}{\partial \phi_3}=a(z_{3,n}),
$$
где $f_n=f(x_n)$ и $z_{i,n}=z_i(x_n)$.

Итого:
$$
\frac{\partial \mathcal L}{\partial \phi_0}=\sum_{n=1}^N r_n,
$$
$$
\frac{\partial \mathcal L}{\partial \phi_1}=\sum_{n=1}^N r_n\,a(z_{1,n}),\quad
\frac{\partial \mathcal L}{\partial \phi_2}=\sum_{n=1}^N r_n\,a(z_{2,n}),\quad
\frac{\partial \mathcal L}{\partial \phi_3}=\sum_{n=1}^N r_n\,a(z_{3,n}).
$$

---

## Градиенты по $\theta$ (6 параметров)
Сначала производные $f_n$:

Для первого скрытого нейрона:
$$
\frac{\partial f_n}{\partial \theta_{10}}=\phi_1\,a'(z_{1,n})\cdot 1=\phi_1\,g_{1,n},
$$
$$
\frac{\partial f_n}{\partial \theta_{11}}=\phi_1\,a'(z_{1,n})\cdot x_n=\phi_1\,g_{1,n}\,x_n.
$$

Для второго:
$$
\frac{\partial f_n}{\partial \theta_{20}}=\phi_2\,g_{2,n},\qquad
\frac{\partial f_n}{\partial \theta_{21}}=\phi_2\,g_{2,n}\,x_n.
$$

Для третьего:
$$
\frac{\partial f_n}{\partial \theta_{30}}=\phi_3\,g_{3,n},\qquad
\frac{\partial f_n}{\partial \theta_{31}}=\phi_3\,g_{3,n}\,x_n.
$$

Теперь умножаем на $r_n$ и суммируем:

$$
\frac{\partial \mathcal L}{\partial \theta_{10}}=\sum_{n=1}^N r_n\,\phi_1\,g_{1,n},\qquad
\frac{\partial \mathcal L}{\partial \theta_{11}}=\sum_{n=1}^N r_n\,\phi_1\,g_{1,n}\,x_n,
$$
$$
\frac{\partial \mathcal L}{\partial \theta_{20}}=\sum_{n=1}^N r_n\,\phi_2\,g_{2,n},\qquad
\frac{\partial \mathcal L}{\partial \theta_{21}}=\sum_{n=1}^N r_n\,\phi_2\,g_{2,n}\,x_n,
$$
$$
\frac{\partial \mathcal L}{\partial \theta_{30}}=\sum_{n=1}^N r_n\,\phi_3\,g_{3,n},\qquad
\frac{\partial \mathcal L}{\partial \theta_{31}}=\sum_{n=1}^N r_n\,\phi_3\,g_{3,n}\,x_n.
$$

---

<details>
<summary>Если в вашей задаче loss без $\frac12$</summary>

Если вместо $\mathcal L=\frac12\sum (f-y)^2$ используется $\tilde{\mathcal L}=\sum (f-y)^2$, то все производные выше умножаются на $2$:
$$
\frac{\partial \tilde{\mathcal L}}{\partial p}=2\,\frac{\partial \mathcal L}{\partial p}.
$$
</details>