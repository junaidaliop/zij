# signProx

Implements signProx, a one-bit proximal algorithm for nonconvex stochastic composite optimization.

signProx minimizes $f(\theta) = d(\theta) + r(\theta)$, where $d$ is a smooth (possibly nonconvex) data-fidelity term and $r$ is a regularizer with a tractable proximal operator. It builds on the stochastic proximal-gradient method, which forms the gradient mapping $\theta - \mathrm{prox}_{\gamma r}(\theta - \gamma\nabla d(\theta))$ as a descent direction. To slash the communication cost of distributed and quantized training, signProx transmits only the sign of that mapping, so each coordinate of the update is encoded in a single bit.

Each step takes a mini-batch stochastic proximal-gradient mapping $\widehat{\mathrm{P}}(\theta) = \frac{1}{B}\sum_{b=1}^{B}\mathrm{prox}_{\gamma r_{k_b}}\!\big(\theta - \gamma\nabla d(\theta)\big)$, subtracts it from the current iterate to obtain the (scaled) gradient mapping direction, and moves a fixed step $\gamma$ against its element-wise sign.

$$
\begin{aligned}
\widehat{\mathrm{P}}(\theta_{t-1}) &= \frac{1}{B}\sum_{b=1}^{B}\mathrm{prox}_{\gamma r_{k_b}}\!\big(\theta_{t-1} - \gamma\, g_{t-1}\big) \\
\theta_t &= \theta_{t-1} - \gamma\,\mathrm{sgn}\!\big(\theta_{t-1} - \widehat{\mathrm{P}}(\theta_{t-1})\big)
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ is the step size, $g_{t-1} = \nabla d(\theta_{t-1})$ is the gradient of the data-fidelity term, $B$ is the mini-batch size, $k_b$ are i.i.d. component indices sampled from the regularizer terms $r_k$, $\mathrm{prox}_{\gamma r_k}$ is the proximal operator of $\gamma r_k$, and $\mathrm{sgn}(\cdot)$ is the element-wise sign that compresses the update to one bit per coordinate.

Reference: Xiaojian Xu, Ulugbek S. Kamilov, "signProx: One-Bit Proximal Algorithm for Nonconvex Stochastic Optimization", arXiv 2018. https://arxiv.org/abs/1807.08023

---
[Back to the Canon](../index.md)
