# A(DP)²SGD

Implements A(DP)²SGD, asynchronous decentralized parallel SGD with differential privacy.

A(DP)²SGD combines the AD-PSGD communication scheme with the Gaussian mechanism. Each of the $K$ workers keeps a local model and computes a mini-batch gradient, then perturbs it with Gaussian noise to provide differential privacy. Instead of a central server, models are mixed across neighbors through a doubly stochastic matrix (gossip averaging), after which each worker takes a local descent step with its noised gradient. The noise variance is set from the privacy budget, the Lipschitz bound, and the iteration count rather than from per-example clipping, so the published update has no explicit clip operation.

$$
\begin{aligned}
\tilde{g}_k^t &= \sum_{i=1}^{B} \nabla F_k(\hat{w}_k^t; \xi_k^{t,i}) + n, \qquad n \sim \mathcal{N}(0, \sigma^2 I) \\
[\,w_1^{t+1/2}, \dots, w_K^{t+1/2}\,] &= [\,w_1^{t}, \dots, w_K^{t}\,]\, A_t \\
w_k^{t+1} &= w_k^{t+1/2} - \eta\, \tilde{g}_k^t
\end{aligned}
$$

where $w_k^t$ is worker $k$'s model, $\hat{w}_k^t$ the (possibly stale) model read for the gradient, $B$ the batch size, $\eta$ the learning rate, $A_t$ a doubly stochastic mixing matrix over the $K$ workers, and $\sigma^2$ the Gaussian noise variance chosen from the privacy budget $(\epsilon, \delta)$.

Reference: Jie Xu, Wei Zhang, Fei Wang, "A(DP)²SGD: Asynchronous Decentralized Parallel Stochastic Gradient Descent with Differential Privacy", arXiv 2020. https://arxiv.org/abs/2008.09246

---
[Back to the Canon](../README.md)
