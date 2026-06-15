# FZOO

Implements FZOO, a fast zeroth-order optimizer that estimates gradients from forward passes alone and adapts its step size by the standard deviation of the perturbed batch losses.

FZOO replaces backpropagation with a batched one-sided difference estimate. At each step it draws $N$ Rademacher random vectors $u_i \in \{\pm 1\}^d$, perturbs the parameters by a radius $\epsilon$, and combines the loss differences $l_i - l_0$ into a single gradient estimate. Dividing the update by the standard deviation $\sigma_t$ of the batch losses yields larger steps in flat regions and smaller steps in steep ones, mirroring Adam-style adaptivity while keeping memory at the inference level. The paper proves this update is formally equivalent to a zeroth-order extension of normalized-SGD.

$$
\begin{aligned}
l_i &= L(\theta_t + \epsilon u_i; B_t), \quad l_0 = L(\theta_t; B_t), \\
g_t &= \frac{1}{\epsilon N} \sum_{i=1}^{N} (l_i - l_0)\, u_i, \\
\sigma_t^2 &= \frac{1}{N-1} \sum_{i=1}^{N} \left( l_i - \frac{1}{N} \sum_{j=1}^{N} l_j \right)^2, \\
\theta_{t+1} &= \theta_t - \eta_t \frac{g_t}{\sigma_t}.
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta_t$ the step size, $\epsilon$ the perturbation radius, $u_i$ the $N$ i.i.d. Rademacher perturbation vectors, $L(\cdot; B_t)$ the loss on mini-batch $B_t$, $g_t$ the one-sided gradient estimate, and $\sigma_t$ the standard deviation of the $N$ perturbed losses.

Reference: Sizhe Dang, Yangyang Guo, Yanjun Zhao, Haishan Ye, Xiaodong Zheng, Guang Dai, Ivor Tsang, "FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed", arXiv 2025. https://arxiv.org/abs/2506.09034

---
[Back to the Canon](../README.md)
