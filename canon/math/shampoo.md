# Shampoo

Implements Shampoo, preconditioned stochastic tensor optimization.

For a matrix parameter $W$ with gradient $G_t$, Shampoo keeps a
left preconditioner $L_t$ over the rows and a right preconditioner
$R_t$ over the columns, each accumulated from the gradient outer
products, and conditions the update on both sides:


$$
\begin{aligned}
L_t &= L_{t-1} + G_t G_t^\top \\
R_t &= R_{t-1} + G_t^\top G_t \\
W_{t+1} &= W_t - \eta\, L_t^{-1/2}\, G_t\, R_t^{-1/2}
\end{aligned}
$$

For a general order-$k$ tensor a preconditioner is maintained for
every dimension by contracting the gradient over the remaining axes, and the
inverse root applied per dimension uses exponent $-1/k$.


**Note:** the original paper (Algorithm 1, matrix case) applies the exponent

$-1/4$ to each preconditioner, giving
$W_{t+1} = W_t - \eta\, L_t^{-1/4} G_t R_t^{-1/4}$. This
implementation instead raises each preconditioner to $-1/k$ for an
order-$k$ tensor (so $-1/2$ for matrices), and recomputes the
inverse roots every `preconditioning_compute_steps` steps.

Reference: Vineet Gupta, Tomer Koren, Yoram Singer, "Shampoo:
Preconditioned Stochastic Tensor Optimization", ICML 2018.
https://arxiv.org/abs/1802.09568

---
[Back to the Canon](../README.md)
