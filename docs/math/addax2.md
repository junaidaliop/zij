# Addax

Implements Addax, a memory-efficient fine-tuning method that mixes zeroth-order and first-order gradient estimates.

Addax computes a true first-order gradient on one minibatch using in-place SGD (where each layer's gradient is consumed and discarded right after it is produced, so the full gradient is never materialized) and a zeroth-order estimate on another minibatch via the SPSA finite-difference rule along a random direction $z$. The two estimates are blended by a single coefficient $\alpha$: the zeroth-order term cuts memory while the first-order term recovers the convergence speed and accuracy that pure zeroth-order methods like MeZO lack.

$$
\begin{aligned}
g^{0}_t &= \frac{1}{|\mathcal{B}^{0}|}\sum_{x \in \mathcal{B}^{0}} \frac{\ell(\theta_t + \epsilon z; x) - \ell(\theta_t - \epsilon z; x)}{2\epsilon}, \quad z \sim \mathcal{N}(0, I) \\
g^{1}_t &= \frac{1}{|\mathcal{B}^{1}|}\sum_{x \in \mathcal{B}^{1}} \nabla \ell(\theta_t; x) \\
\theta_{t+1} &= \theta_t - \eta\left( \alpha\, z\, g^{0}_t + (1-\alpha)\, g^{1}_t \right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g^{0}_t$ the scalar zeroth-order directional derivative along the random direction $z$, $g^{1}_t$ the first-order gradient, $\epsilon$ the perturbation scale, $\mathcal{B}^{0}$ and $\mathcal{B}^{1}$ the zeroth-order and first-order minibatches, and $\alpha \in [0,1]$ the coefficient balancing the two estimates.

Reference: Zeman Li, Xinwei Zhang, Peilin Zhong, Yuan Deng, Meisam Razaviyayn, Vahab Mirrokni, "Addax: Utilizing Zeroth-Order Gradients to Improve Memory Efficiency and Performance of SGD for Fine-Tuning Language Models", ICLR 2025. https://arxiv.org/abs/2410.06441

---
[Back to the Canon](../index.md)
