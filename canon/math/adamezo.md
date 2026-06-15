# AdaMeZO

Implements AdaMeZO, an Adam-style zeroth-order optimizer that recovers first- and second-moment estimates without storing them in memory.

AdaMeZO fine-tunes large models using only forward passes. As in MeZO, the gradient is replaced by the SPSA estimate along a single random Gaussian direction $z_t$, giving a rank-1 reconstruction that costs two forward passes per step. To gain Adam's curvature awareness without tripling memory, AdaMeZO does not keep the moments $m_t,v_t$ in memory. Instead it caches only the per-step random seeds (or PRNG states) and the scalar projected gradients, then unrolls the exponential moving averages into a truncated sum over a finite horizon $h$: gradients older than $h$ steps are discarded, and the surviving terms are regenerated on the fly by replaying the cached random streams. The truncated moments are reconstructed block-wise so the model can be updated in place.

$$
\begin{aligned}
g_t &= \frac{\mathcal{L}(\theta_{t-1} + \mu z_t; B_t) - \mathcal{L}(\theta_{t-1} - \mu z_t; B_t)}{2\mu}\, z_t, \qquad z_t \sim \mathcal{N}(0, I_d) \\
m_t &\approx (1 - \beta_1)\big(g_t + \beta_1 g_{t-1} + \cdots + \beta_1^{\,t-h-1} g_{t-h-1}\big) \\
v_t &\approx (1 - \beta_2)\big(g_t \odot g_t + \beta_2\, g_{t-1} \odot g_{t-1} + \cdots + \beta_2^{\,t-h-1} g_{t-h-1} \odot g_{t-h-1}\big) \\
\theta_t &= \theta_{t-1} - \eta\, \frac{m_t}{\sqrt{v_t + \epsilon}}
\end{aligned}
$$

where $\theta \in \mathbb{R}^d$ are the parameters, $\mathcal{L}(\cdot; B_t)$ is the loss on minibatch $B_t$, $z_t$ is an i.i.d. standard Gaussian perturbation, $\mu$ is the perturbation scale, $\eta$ is the learning rate, $g_t$ is the SPSA gradient estimate (the scalar finite-difference quotient times $z_t$), $\odot$ is the elementwise product, $\beta_1,\beta_2 \in (0,1)$ are the moment decay rates, $h$ is the finite moment horizon beyond which old gradients are truncated, and $\epsilon$ is a small stability constant. The moments are never stored; they are recomputed each step by replaying cached seeds and projected gradients.

Reference: Zhijie Cai, Haolong Chen, Guangxu Zhu, "AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments", arXiv 2026. https://arxiv.org/abs/2605.00650

---
[Back to the Canon](../README.md)
