# AdaS

Implements AdaS, a per-layer learning rate scheduler driven by the knowledge gain accumulated in each layer's weights.

AdaS measures, after each epoch, how much useful low-rank structure a layer has acquired by decomposing its weight tensor and summing the normalized singular values into a "knowledge gain" $\bar G$. The per-layer learning rate is then updated as an exponential moving average that grows with the gain still being earned and decays once a layer's gain saturates, so layers that keep learning retain a larger step size while converged layers are slowed. The resulting rate feeds an otherwise standard SGD-with-momentum update.

$$
\begin{aligned}
\bar G(t,\ell) &= \tfrac{1}{2}\big[G_{3}(t,\ell) + G_{4}(t,\ell)\big] \\
\eta(t,\ell) &= \beta\,\eta(t-1,\ell) + \zeta\,\big[\bar G(t,\ell) - \bar G(t-1,\ell)\big] \\
v_t &= \alpha\,v_{t-1} - \eta(t,\ell)\,g_t \\
\theta_t &= \theta_{t-1} + v_t
\end{aligned}
$$

where $\ell$ indexes a layer, $t$ the epoch, $\bar G(t,\ell)$ the average knowledge gain over the weight tensor's mode-3 and mode-4 unfoldings, $\beta$ the gain-factor (EMA decay), $\zeta$ the knowledge-gain scaling (set to $1$), $\alpha$ the momentum coefficient, $g_t$ the gradient, $v_t$ the velocity, and $\theta$ the parameters. Each $G$ is $G_{d}=\tfrac{1}{N_d\,\sigma_1}\sum_i \sigma_i$ over the singular values $\sigma_1\ge\sigma_2\ge\cdots$ of the unfolded weights.

Reference: Mahdi S. Hosseini, Konstantinos N. Plataniotis, "AdaS: Adaptive Scheduling of Stochastic Gradients", arXiv 2020. https://arxiv.org/abs/2006.06587

---
[Back to the Canon](../index.md)
