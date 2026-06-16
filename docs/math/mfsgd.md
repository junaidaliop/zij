# mF-SGD

Implements mF-SGD, a momentum-accelerated fractional stochastic gradient descent for matrix-factorization recommender systems.

The method factorizes the rating matrix into user features $a_u$ and item features $b_i$ by minimizing the squared prediction error. Beyond the ordinary gradient, it adds a Riemann–Liouville fractional-order gradient term: the fractional derivative of the squared error contributes a factor $|\theta|^{1-\nu}/\Gamma(2-\nu)$, giving an extra search direction controlled by the fractional order $\nu$. A momentum (velocity) accumulation over the combined integer- and fractional-order gradients then accelerates convergence relative to plain fractional SGD.

For each observed entry the error is $E_{ui} = C_{ui} - a_u^\top b_i$. Writing $g_t$ for the combined gradient and $v_t$ for the velocity, the per-feature updates are

$$
\begin{aligned}
g^{(a)}_t &= \eta\, E_{ui}\, b_i + \frac{\eta_{fr}}{\Gamma(2-\nu)}\, E_{ui}\, b_i \odot |a_u|^{\,1-\nu}, \\
g^{(b)}_t &= \eta\, E_{ui}\, a_u + \frac{\eta_{fr}}{\Gamma(2-\nu)}\, E_{ui}\, a_u \odot |b_i|^{\,1-\nu}, \\
v^{(a)}_t &= \beta\, v^{(a)}_{t-1} + g^{(a)}_t, \qquad a_u \leftarrow a_u + v^{(a)}_t, \\
v^{(b)}_t &= \beta\, v^{(b)}_{t-1} + g^{(b)}_t, \qquad b_i \leftarrow b_i + v^{(b)}_t,
\end{aligned}
$$

where $\eta$ is the integer-order learning rate, $\eta_{fr}$ the fractional learning rate, $\nu \in (0,1)$ the fractional order, $\Gamma(\cdot)$ the Gamma function, $\beta \in (0,1)$ the momentum weight, $\odot$ elementwise product, and $E_{ui}$ the prediction error on entry $(u,i)$. The signs are additive because the error gradient $\partial E_{ui}^2$ is negated toward the minimum.

Reference: Zeshan Aslam Khan, Syed Zubair, Hani Alquhayz, Muhammad Azeem, Allah Ditta, "Design of Momentum Fractional Stochastic Gradient Descent for Recommender Systems", IEEE Access 2019. https://doi.org/10.1109/ACCESS.2019.2954859

---
[Back to the Canon](../index.md)
