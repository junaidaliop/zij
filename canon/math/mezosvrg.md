# MeZO-SVRG

Implements MeZO-SVRG, a memory-efficient zeroth-order optimizer that couples forward-pass gradient estimation with SVRG variance reduction.

Like MeZO, gradients are estimated through forward passes only, using a two-point (SPSA) finite-difference estimate along a shared random direction $z$, so no backpropagation memory is required. To curb the high variance of single zeroth-order estimates, MeZO-SVRG periodically computes a full-batch estimate $g$ at a snapshot point $\bar\theta$ (every $q$ steps) and corrects the noisy minibatch estimate on the intermediate steps with the SVRG control variate $\bar\nabla f_{\mathcal I_t}(\theta_t) - \bar\nabla f_{\mathcal I_t}(\bar\theta) + g$. A larger step size $\eta_1$ is used on the full-batch steps and a smaller $\eta_2$ on the variance-reduced minibatch steps.

$$
\begin{aligned}
\bar\nabla f_{\mathcal I}(\theta) &= \frac{1}{2\mu}\left(\frac{1}{b}\sum_{i\in\mathcal I}\big[f_i(\theta+\mu z)-f_i(\theta-\mu z)\big]\right) z, \qquad z\sim\mathcal N(0,I) \\
\text{every } q \text{ steps:}\quad g &\leftarrow \bar\nabla f(\theta_t), \quad \bar\theta \leftarrow \theta_t, \quad \theta_{t+1} = \theta_t - \eta_1\, g \\
\text{otherwise:}\quad \theta_{t+1} &= \theta_t - \eta_2\big[\bar\nabla f_{\mathcal I_t}(\theta_t) - \bar\nabla f_{\mathcal I_t}(\bar\theta) + g\big]
\end{aligned}
$$

where $\theta$ are the parameters, $\mu$ the perturbation scale, $z$ a fresh standard-normal direction per estimate, $f_i$ the per-sample loss, $\mathcal I_t$ a minibatch of size $b$, $\bar\nabla f$ the full-batch estimate over all $n$ samples, $g$ the snapshot full-batch gradient, $\bar\theta$ the snapshot parameters, $q$ the full-batch period, and $\eta_1>\eta_2$ the two learning rates.

Reference: Tanmay Gautam, Youngsuk Park, Hao Zhou, Parameswaran Raman, Wooseok Ha, "Variance-reduced Zeroth-Order Methods for Fine-Tuning Language Models", ICML 2024. https://arxiv.org/abs/2404.08080

---
[Back to the Canon](../README.md)
