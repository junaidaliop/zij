# AdamP

Implements AdamP, Adam with a scale-invariant projection step.

For each layer-weight parameter, the Adam update $p_t$ is split into
its radial and tangential components relative to the weight $\theta$,
and the radial part is removed whenever the cosine similarity between the
gradient and the weight is below a threshold (i.e. the weight is treated as
scale-invariant):


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
     p_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
     q_t &= \Pi_{\theta_{t-1}}(p_t) \\
     \theta_t &= \theta_{t-1} - \eta \, q_t
\end{aligned}
$$

where $\Pi_{\theta}(p) = p - (\hat{\theta} \cdot p)\,\hat{\theta}$
projects out the component of $p$ along the unit weight
$\hat{\theta}$ and `wd_ratio` scales the decoupled weight decay on
the projected parameters.

Reference: Byeongho Heo, Sanghyuk Chun, Seong Joon Oh, Dongyoon Han,
Sangdoo Yun, Gyuwan Kim, Youngjung Uh, Jung-Woo Ha, "AdamP: Slowing Down
the Slowdown for Momentum Optimizers on Scale-invariant Weights", ICLR 2021.
https://arxiv.org/abs/2006.08217

---
[Back to the Canon](../index.md)
