Обозначим:

$$
z_1=\theta_{01}+\theta_{11}x,\qquad z_2=\theta_{02}+\theta_{12}x
$$

$$
h_1=a[z_1],\qquad h_2=a[z_2]
$$

$$
u_1=\psi_{01}+\psi_{11}h_1+\psi_{21}h_2,\qquad
u_2=\psi_{02}+\psi_{12}h_1+\psi_{22}h_2
$$

$$
g_1=a[u_1],\qquad g_2=a[u_2]
$$

$$
y=\phi_0+\phi_1 g_1+\phi_2 g_2
$$

И производная ReLU:

$$
\frac{\partial a[z]}{\partial z}=I[z>0].
$$

---

## 1) Производные по $\phi$

$$
\frac{\partial y}{\partial \phi_0}=1
$$

$$
\frac{\partial y}{\partial \phi_1}=g_1=a[u_1]
$$

$$
\frac{\partial y}{\partial \phi_2}=g_2=a[u_2]
$$

---

## 2) Производные по $\psi$

Для параметров, входящих в $u_1$:

$$
\frac{\partial y}{\partial \psi_{01}}=\phi_1\,I[u_1>0]
$$

$$
\frac{\partial y}{\partial \psi_{11}}=\phi_1\,I[u_1>0]\;h_1
$$

$$
\frac{\partial y}{\partial \psi_{21}}=\phi_1\,I[u_1>0]\;h_2
$$

Для параметров, входящих в $u_2$:

$$
\frac{\partial y}{\partial \psi_{02}}=\phi_2\,I[u_2>0]
$$

$$
\frac{\partial y}{\partial \psi_{12}}=\phi_2\,I[u_2>0]\;h_1
$$

$$
\frac{\partial y}{\partial \psi_{22}}=\phi_2\,I[u_2>0]\;h_2
$$

---

## 3) Производные по $\theta$

Сначала выпишем производные $y$ по $h_1,h_2$:

$$
\frac{\partial y}{\partial h_1}
=\phi_1 I[u_1>0]\psi_{11}+\phi_2 I[u_2>0]\psi_{12}
$$

$$
\frac{\partial y}{\partial h_2}
=\phi_1 I[u_1>0]\psi_{21}+\phi_2 I[u_2>0]\psi_{22}
$$

Далее производные $h_1,h_2$ по $\theta$:

$$
\frac{\partial h_1}{\partial \theta_{01}}=I[z_1>0],\qquad
\frac{\partial h_1}{\partial \theta_{11}}=I[z_1>0]\;x
$$

$$
\frac{\partial h_2}{\partial \theta_{02}}=I[z_2>0],\qquad
\frac{\partial h_2}{\partial \theta_{12}}=I[z_2>0]\;x
$$

Итого (по правилу цепочки):

$$
\frac{\partial y}{\partial \theta_{01}}
=\Big(\phi_1 I[u_1>0]\psi_{11}+\phi_2 I[u_2>0]\psi_{12}\Big)\;I[z_1>0]
$$

$$
\frac{\partial y}{\partial \theta_{11}}
=\Big(\phi_1 I[u_1>0]\psi_{11}+\phi_2 I[u_2>0]\psi_{12}\Big)\;I[z_1>0]\;x
$$

$$
\frac{\partial y}{\partial \theta_{02}}
=\Big(\phi_1 I[u_1>0]\psi_{21}+\phi_2 I[u_2>0]\psi_{22}\Big)\;I[z_2>0]
$$

$$
\frac{\partial y}{\partial \theta_{12}}
=\Big(\phi_1 I[u_1>0]\psi_{21}+\phi_2 I[u_2>0]\psi_{22}\Big)\;I[z_2>0]\;x
$$

---