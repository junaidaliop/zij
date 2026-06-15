# ZO2

Implements ZO2, zeroth-order fine-tuning of large language models with CPU offloading.

ZO2 estimates gradients without backpropagation using the randomized gradient estimator (RGE): two forward passes at parameters perturbed along a single random direction $z$ yield a scalar finite-difference estimate of the directional derivative, which is then projected back to give the full gradient estimate $g_t z$. Because no activations or backward graph are stored, optimizer and parameter states can be offloaded to CPU memory, letting extremely large models fine-tune under tight GPU budgets. The same random direction $z$ must be reused for both the loss evaluations and the update step.

$$
\begin{aligned}
g_t &= \frac{\mathcal{L}(\theta_t + \epsilon z) - \mathcal{L}(\theta_t - \epsilon z)}{2\epsilon}, \\
\theta_{t+1} &= \theta_t - \eta\, g_t\, z.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $z \sim \mathcal{N}(0, I)$ a random perturbation direction, $\epsilon > 0$ the smoothing scale, $\mathcal{L}$ the loss, and $g_t$ the scalar projected gradient estimate.

Reference: Liangyu Wang, Jie Ren, Hang Xu, Junxiao Wang, Huanyi Xie, David E. Keyes, Di Wang, "ZO2: Scalable Zeroth-Order Fine-Tuning for Extremely Large Language Models with Limited GPU Memory", arXiv 2025. https://arxiv.org/abs/2503.12668

---
[Back to the Canon](../README.md)
