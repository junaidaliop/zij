# RAdamScheduleFree

Implements Schedule-Free RAdam, which needs neither a schedule nor a warmup period.

The update follows `AdamWScheduleFree` with the step size scaled by
the RAdam rectification term in place of explicit warmup:


$$
\begin{aligned}
    \rho_\infty &= \frac{2}{1 - \beta_2} - 1, \qquad
        \rho_t = \rho_\infty - \frac{2 t \beta_2^t}{1 - \beta_2^t}, \\
    \gamma_t &= \gamma
        \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                   {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}}
        \quad \text{if } \rho_t > 4,
\end{aligned}
$$

while for $\rho_t \le 4$ the second-moment normalization is skipped
and the step degenerates to SGD (or, with `silent_sgd_phase`, only the
moment estimates are updated).

Reference: Aaron Defazio, Xingyu Yang, Harsh Mehta, Konstantin Mishchenko,
Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
https://arxiv.org/abs/2405.15682
Rectification: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen,
Xiaodong Liu, Jianfeng Gao, Jiawei Han, "On the Variance of the Adaptive
Learning Rate and Beyond", ICLR 2020.
https://arxiv.org/abs/1908.03265


**Note:** Call `optimizer.train()` before training and `optimizer.eval()` before evaluation or checkpointing, alongside the matching `model.train()` / `model.eval()` calls. Gradients are computed at $y_t$ while losses should be measured at $x_t$, so the parameter buffer must be switched between the two points.


---
[Back to the Canon](../index.md)
