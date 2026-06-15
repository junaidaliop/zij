# KerZOO

Implements KerZOO, a kernel-informed zeroth-order optimizer for memory-efficient LLM fine-tuning.

Standard zeroth-order methods estimate gradients from finite-difference function queries along random directions, but the symmetric two-point estimator carries a leading bias from third-order terms in the Taylor expansion. KerZOO injects a random scalar $r$ into the perturbation and reweights each finite difference by a polynomial kernel $K(r)$ chosen so that its first moment is preserved while its third moment vanishes, canceling the dominant bias term and accelerating convergence without storing any first-order gradients.

$$
\begin{aligned}
\hat{g}_t &= \frac{1}{n}\sum_{i=1}^{n} \frac{\mathcal{L}(\theta_t + \epsilon r_i u_i) - \mathcal{L}(\theta_t - \epsilon r_i u_i)}{2\epsilon}\, K(r_i)\, u_i, \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t, \\
\text{with}\quad K(r) &= C\cdot\tfrac{15}{4}\, r\,(5 - 7 r^2), \quad \mathbb{E}[r K(r)] = C,\ \ \mathbb{E}[r^3 K(r)] = 0.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\epsilon$ the perturbation scale, $u_i \sim \mathcal{N}(0, I_d)$ the random directions, $r_i \sim \mathrm{Uniform}[-1,1]$ the random scalars, $n$ the number of perturbations per step, $\mathcal{L}$ the (minibatch) loss, and $K$ the bias-canceling kernel ($K(r)=3Cr$ and $K(r)=C\tfrac{195}{64}r(99r^4-126r^2+35)$ are the lower- and higher-order alternatives).

Reference: Zhendong Mi, Qitao Tan, Xiaodong Yu, Zining Zhu, Geng Yuan, Shaoyi Huang, "KerZOO: Kernel Function Informed Zeroth-Order Optimization for Accurate and Accelerated LLM Fine-Tuning", arXiv 2025. https://arxiv.org/abs/2505.18886

---
[Back to the Canon](../README.md)
