# GRZO

Implements GRZO, a group-relative zeroth-order optimizer for memory-efficient LLM fine-tuning.

GRZO removes the high variance of single-direction zeroth-order methods like MeZO by drawing one pseudo-independent perturbation $z_i$ per mini-batch example instead of a single shared direction. The perturbations are built cheaply through a Flipout-style sign factorization, so all $B$ directions share one base tensor and the step keeps MeZO's two-forward-pass budget and inference-level memory.

Each example contributes a two-sided loss difference $\delta_i$, and these are turned into advantage-like weights through GRPO-style group-relative normalization: dividing by the within-batch standard deviation makes the update scale-invariant to loss magnitude and acts as an adaptive effective step size. The normalized weights then scale their respective perturbation directions into the gradient estimate.

$$
\begin{aligned}
\delta_i &= L(\theta_t + \sigma z_i;\, \xi_i) - L(\theta_t - \sigma z_i;\, \xi_i) \\
\bar{\delta} &= \frac{1}{B}\sum_{i=1}^{B}\delta_i, \qquad s = \sqrt{\frac{1}{B}\sum_{i=1}^{B}(\delta_i - \bar{\delta})^2} \\
a_i &= \frac{\delta_i}{s + \epsilon} \\
\hat{g}_t &= \frac{1}{2\sigma B}\sum_{i=1}^{B} a_i\, z_i \\
\theta_{t+1} &= \theta_t - \eta\, \hat{g}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $z_i$ the per-example perturbation direction $\mathrm{vec}(U \odot (r_i s_i^\top))$ with shared base $U$ and Rademacher sign vectors $r_i, s_i$, $\sigma$ the perturbation scale, $\xi_i$ the $i$-th example, $\delta_i$ its two-sided loss difference, $\bar{\delta}$ and $s$ the within-batch mean and standard deviation of the $\delta_i$, $a_i$ the group-relative weights, $B$ the batch size, $\hat{g}_t$ the gradient estimator, and $\epsilon$ a small stability constant. The factor $1/(s+\epsilon)$ behaves as an adaptive effective step size $\tilde{\eta}_t = \eta/(s_t + \epsilon)$.

Reference: Liyan Tan, Yequan Zhao, Yifan Yang, Ruijie Zhang, Xinling Yu, Zheng Zhang, "GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning", arXiv 2026. https://arxiv.org/abs/2606.02857

---
[Back to the Canon](../README.md)
